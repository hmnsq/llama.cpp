#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#அந்தர்பாவ <string.h>
#அந்தர்பாவ <math.h>

#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"

#define sum_rows_preamble                         \
    மாறிலி கட்டமைப்பு htp_tensor *src0 = octx->src[0]; \
    மாறிலி கட்டமைப்பு htp_tensor *dst  = octx->dst;    \
                                                  \
    மாறிலி uint32_t ne00 = src0->ne[0];            \
    மாறிலி uint32_t ne01 = src0->ne[1];            \
    மாறிலி uint32_t ne02 = src0->ne[2];            \
    மாறிலி uint32_t ne03 = src0->ne[3];            \
                                                  \
    மாறிலி uint32_t nb00 = src0->nb[0];            \
    மாறிலி uint32_t nb01 = src0->nb[1];            \
    மாறிலி uint32_t nb02 = src0->nb[2];            \
    மாறிலி uint32_t nb03 = src0->nb[3];            \
                                                  \
    மாறிலி uint32_t  ne0 = dst->ne[0];             \
    மாறிலி uint32_t  ne1 = dst->ne[1];             \
    மாறிலி uint32_t  ne2 = dst->ne[2];             \
    மாறிலி uint32_t  ne3 = dst->ne[3];             \
                                                  \
    மாறிலி uint32_t  nb0 = dst->nb[0];             \
    மாறிலி uint32_t  nb1 = dst->nb[1];             \
    மாறிலி uint32_t  nb2 = dst->nb[2];             \
    மாறிலி uint32_t  nb3 = dst->nb[3];             \

கட்டமைப்பு sum_rows_context {
    கட்டமைப்பு htp_ops_context * octx;
    மாறிலி uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

நிலையான வெறுமை sum_rows_thread_f32(unsigned முழு nth, unsigned முழு ith, வெறுமை *data) {
    மாறிலி கட்டமைப்பு sum_rows_context * smctx = (மாறிலி கட்டமைப்பு sum_rows_context *) data;

    மாறிலி uint32_t rows_per_thread = smctx->rows_per_thread;
    மாறிலி uint32_t total_rows      = smctx->total_rows;

    மாறிலி uint32_t start_row = rows_per_thread * ith;
    மாறிலி uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    எனில் (start_row >= end_row) {
        திரும்பு;
    }

    மாறிலி size_t   src_stride = smctx->src_stride;
    மாறிலி size_t   dst_stride = smctx->dst_stride;
    மாறிலி uint32_t ne00       = smctx->ne00;
    மாறிலி bool     opt_path   = smctx->opt_path;

    மாறிலி தசம * restrict src_th = (மாறிலி தசம *) (smctx->src_data + (start_row * src_stride));
    தசம       * restrict dst_th = (தசம *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    மாறிலி uint32_t n_rows = end_row - start_row;

    கட்டமைப்பு htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    க்கு (uint32_t ir = 0; ir < n_rows; ir++) {
        மாறிலி தசம * restrict src_local = (மாறிலி தசம *) ((மாறிலி uint8_t *) src_th + ir * src_stride);
        தசம       * restrict dst_local = (தசம *)       ((uint8_t *)       dst_th + ir * dst_stride);

        எனில் (ir + 1 < n_rows) {
            hex_l2fetch((மாறிலி uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        எனில் (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((மாறிலி uint8_t *) src_local, ne00);
        } இல்லையெனில் {
            *dst_local = hvx_reduce_sum_f32((மாறிலி uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

முழு op_sum_rows(கட்டமைப்பு htp_ops_context * octx) {
    sum_rows_preamble;

    எனில் (octx->src[0]->type != HTP_TYPE_F32) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி uint32_t src0_nrows      = ne01 * ne02 * ne03;
    மாறிலி size_t dst_data_row_size = dst->ne[0] * அளவு(தசம);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, அளவு(தசம), (uint32_t) dst_data_row_size, &rows_per_chunk);
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
    }

    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t n_threads = octx->n_threads;
    மாறிலி uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    எனில் ((0 == hex_is_aligned((வெறுமை *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    கட்டமைப்பு sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (மாறிலி uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    திரும்பு HTP_STATUS_OK;
}

கட்டமைப்பு sum_context {
    கட்டமைப்பு htp_ops_context * octx;
    மாறிலி தசம            * src_data;
    தசம                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

நிலையான வெறுமை sum_thread_f32(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு sum_context * sctx = (கட்டமைப்பு sum_context *) data;
    மாறிலி uint32_t start = sctx->elems_per_thread * ith;
    மாறிலி uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    எனில் (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        திரும்பு;
    }

    மாறிலி uint32_t n = end - start;
    மாறிலி தசம * src = sctx->src_data + start;

    கட்டமைப்பு htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((மாறிலி வெறுமை *) src, n * அளவு(தசம));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((மாறிலி uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

முழு op_sum(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    எனில் (src0->type != HTP_TYPE_F32) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (octx->ctx->mdev.மொத்தம் > 1 && octx->ctx->mdev.idx > 0) {
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    எனில் (total_elems == 0) {
        ((தசம *) dst->data)[0] = 0.0f;
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    மாறிலி uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    மாறிலி uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    கட்டமைப்பு sum_context sctx = {
        .octx             = octx,
        .src_data         = (மாறிலி தசம *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    தசம sum = 0.0f;
    க்கு (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((தசம *) dst->data)[0] = sum;

    திரும்பு HTP_STATUS_OK;
}

நிலையான உள்வரி வெறுமை argmax_slice_f32(
    மாறிலி தசம * restrict src,
    uint32_t n,
    uint32_t offset,
    தசம * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

கட்டமைப்பு argmax_context {
    கட்டமைப்பு htp_ops_context * octx;
    மாறிலி தசம            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    தசம                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

நிலையான வெறுமை argmax_thread_single_row(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு argmax_context * actx = (கட்டமைப்பு argmax_context *) data;
    மாறிலி uint32_t start = actx->elems_per_thread * ith;
    மாறிலி uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    எனில் (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        திரும்பு;
    }

    மாறிலி uint32_t n = end - start;
    மாறிலி தசம * src = actx->src_data + start;

    கட்டமைப்பு htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((மாறிலி வெறுமை *) src, n * அளவு(தசம));

    தசம max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

நிலையான வெறுமை argmax_thread_multi_row(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு argmax_context * actx = (கட்டமைப்பு argmax_context *) data;
    மாறிலி uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    மாறிலி uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    எனில் (r0 >= r1) {
        திரும்பு;
    }

    கட்டமைப்பு htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    க்கு (uint32_t r = r0; r < r1; r++) {
        மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((மாறிலி வெறுமை *) src_row, actx->ne00 * அளவு(தசம));

        தசம max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

முழு op_argmax(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    எனில் (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி uint32_t ne00 = src0->ne[0];
    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    எனில் (ne00 == 0 || src0_nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    எனில் (src0_nrows == 1) {
        எனில் (octx->ctx->mdev.மொத்தம் > 1 && octx->ctx->mdev.idx > 0) {
            திரும்பு HTP_STATUS_OK;
        }

        எனில் (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            திரும்பு HTP_STATUS_OK;
        }

        மாறிலி uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        மாறிலி uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        மாறிலி uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        கட்டமைப்பு argmax_context actx = {
            .octx             = octx,
            .src_data         = (மாறிலி தசம *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * அளவு(தசம)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) அளவு(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        தசம best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        க்கு (uint32_t i = 1; i < n_threads; i++) {
            எனில் (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        திரும்பு HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        மாறிலி bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, அளவு(int32_t));
        மாறிலி uint32_t elems_per_chunk = can_split ? 32 : 0;
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
    }

    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t n_threads = MIN(octx->n_threads, nrows);
    மாறிலி uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    கட்டமைப்பு argmax_context actx = {
        .octx             = octx,
        .src_data         = (மாறிலி தசம *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * அளவு(தசம)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) அளவு(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    திரும்பு HTP_STATUS_OK;
}
