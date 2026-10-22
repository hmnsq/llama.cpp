#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <string.h>
#شامل <math.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"

#define sum_rows_preamble                         \
    ثابت ساختار htp_tensor *src0 = octx->src[0]; \
    ثابت ساختار htp_tensor *dst  = octx->dst;    \
                                                  \
    ثابت uint32_t ne00 = src0->ne[0];            \
    ثابت uint32_t ne01 = src0->ne[1];            \
    ثابت uint32_t ne02 = src0->ne[2];            \
    ثابت uint32_t ne03 = src0->ne[3];            \
                                                  \
    ثابت uint32_t nb00 = src0->nb[0];            \
    ثابت uint32_t nb01 = src0->nb[1];            \
    ثابت uint32_t nb02 = src0->nb[2];            \
    ثابت uint32_t nb03 = src0->nb[3];            \
                                                  \
    ثابت uint32_t  ne0 = dst->ne[0];             \
    ثابت uint32_t  ne1 = dst->ne[1];             \
    ثابت uint32_t  ne2 = dst->ne[2];             \
    ثابت uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ثابت uint32_t  nb0 = dst->nb[0];             \
    ثابت uint32_t  nb1 = dst->nb[1];             \
    ثابت uint32_t  nb2 = dst->nb[2];             \
    ثابت uint32_t  nb3 = dst->nb[3];             \

ساختار sum_rows_context {
    ساختار htp_ops_context * octx;
    ثابت uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

ایستا تهی sum_rows_thread_f32(unsigned صحیح nth, unsigned صحیح ith, تهی *data) {
    ثابت ساختار sum_rows_context * smctx = (ثابت ساختار sum_rows_context *) data;

    ثابت uint32_t rows_per_thread = smctx->rows_per_thread;
    ثابت uint32_t total_rows      = smctx->total_rows;

    ثابت uint32_t start_row = rows_per_thread * ith;
    ثابت uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    اگر (start_row >= end_row) {
        بازگشت;
    }

    ثابت size_t   src_stride = smctx->src_stride;
    ثابت size_t   dst_stride = smctx->dst_stride;
    ثابت uint32_t ne00       = smctx->ne00;
    ثابت bool     opt_path   = smctx->opt_path;

    ثابت اعشاری * restrict src_th = (ثابت اعشاری *) (smctx->src_data + (start_row * src_stride));
    اعشاری       * restrict dst_th = (اعشاری *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    ثابت uint32_t n_rows = end_row - start_row;

    ساختار htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    برای (uint32_t ir = 0; ir < n_rows; ir++) {
        ثابت اعشاری * restrict src_local = (ثابت اعشاری *) ((ثابت uint8_t *) src_th + ir * src_stride);
        اعشاری       * restrict dst_local = (اعشاری *)       ((uint8_t *)       dst_th + ir * dst_stride);

        اگر (ir + 1 < n_rows) {
            hex_l2fetch((ثابت uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        اگر (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((ثابت uint8_t *) src_local, ne00);
        } وگرنه {
            *dst_local = hvx_reduce_sum_f32((ثابت uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

صحیح op_sum_rows(ساختار htp_ops_context * octx) {
    sum_rows_preamble;

    اگر (octx->src[0]->type != HTP_TYPE_F32) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ثابت size_t dst_data_row_size = dst->ne[0] * اندازه(اعشاری);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.شمار > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, اندازه(اعشاری), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        nrows     = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = octx->n_threads;
    ثابت uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    اگر ((0 == hex_is_aligned((تهی *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    ساختار sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (ثابت uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    بازگشت HTP_STATUS_OK;
}

ساختار sum_context {
    ساختار htp_ops_context * octx;
    ثابت اعشاری            * src_data;
    اعشاری                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

ایستا تهی sum_thread_f32(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    ساختار sum_context * sctx = (ساختار sum_context *) data;
    ثابت uint32_t start = sctx->elems_per_thread * ith;
    ثابت uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    اگر (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        بازگشت;
    }

    ثابت uint32_t n = end - start;
    ثابت اعشاری * src = sctx->src_data + start;

    ساختار htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ثابت تهی *) src, n * اندازه(اعشاری));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((ثابت uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

صحیح op_sum(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    اگر (src0->type != HTP_TYPE_F32) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    اگر (octx->ctx->mdev.شمار > 1 && octx->ctx->mdev.idx > 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    اگر (total_elems == 0) {
        ((اعشاری *) dst->data)[0] = 0.0f;
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    ثابت uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    ثابت uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    ساختار sum_context sctx = {
        .octx             = octx,
        .src_data         = (ثابت اعشاری *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    اعشاری sum = 0.0f;
    برای (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((اعشاری *) dst->data)[0] = sum;

    بازگشت HTP_STATUS_OK;
}

ایستا درونخطی تهی argmax_slice_f32(
    ثابت اعشاری * restrict src,
    uint32_t n,
    uint32_t offset,
    اعشاری * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

ساختار argmax_context {
    ساختار htp_ops_context * octx;
    ثابت اعشاری            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    اعشاری                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ایستا تهی argmax_thread_single_row(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    ساختار argmax_context * actx = (ساختار argmax_context *) data;
    ثابت uint32_t start = actx->elems_per_thread * ith;
    ثابت uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    اگر (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        بازگشت;
    }

    ثابت uint32_t n = end - start;
    ثابت اعشاری * src = actx->src_data + start;

    ساختار htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ثابت تهی *) src, n * اندازه(اعشاری));

    اعشاری max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

ایستا تهی argmax_thread_multi_row(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    ساختار argmax_context * actx = (ساختار argmax_context *) data;
    ثابت uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    ثابت uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    اگر (r0 >= r1) {
        بازگشت;
    }

    ساختار htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    برای (uint32_t r = r0; r < r1; r++) {
        ثابت اعشاری * src_row = (ثابت اعشاری *) ((ثابت uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((ثابت تهی *) src_row, actx->ne00 * اندازه(اعشاری));

        اعشاری max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

صحیح op_argmax(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    اگر (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    اگر (ne00 == 0 || src0_nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    اگر (src0_nrows == 1) {
        اگر (octx->ctx->mdev.شمار > 1 && octx->ctx->mdev.idx > 0) {
            بازگشت HTP_STATUS_OK;
        }

        اگر (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            بازگشت HTP_STATUS_OK;
        }

        ثابت uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        ثابت uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        ثابت uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        ساختار argmax_context actx = {
            .octx             = octx,
            .src_data         = (ثابت اعشاری *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * اندازه(اعشاری)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) اندازه(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        اعشاری best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        برای (uint32_t i = 1; i < n_threads; i++) {
            اگر (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        بازگشت HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, اندازه(int32_t));
        ثابت uint32_t elems_per_chunk = can_split ? 32 : 0;
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        nrows     = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = MIN(octx->n_threads, nrows);
    ثابت uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    ساختار argmax_context actx = {
        .octx             = octx,
        .src_data         = (ثابت اعشاری *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * اندازه(اعشاری)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) اندازه(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    بازگشت HTP_STATUS_OK;
}
