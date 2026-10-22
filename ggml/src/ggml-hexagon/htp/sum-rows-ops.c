#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <HAP_perf.h>

#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <math.h>

#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"

#define sum_rows_preamble                         \
    અચલ સંરચના htp_tensor *src0 = octx->src[0]; \
    અચલ સંરચના htp_tensor *dst  = octx->dst;    \
                                                  \
    અચલ uint32_t ne00 = src0->ne[0];            \
    અચલ uint32_t ne01 = src0->ne[1];            \
    અચલ uint32_t ne02 = src0->ne[2];            \
    અચલ uint32_t ne03 = src0->ne[3];            \
                                                  \
    અચલ uint32_t nb00 = src0->nb[0];            \
    અચલ uint32_t nb01 = src0->nb[1];            \
    અચલ uint32_t nb02 = src0->nb[2];            \
    અચલ uint32_t nb03 = src0->nb[3];            \
                                                  \
    અચલ uint32_t  ne0 = dst->ne[0];             \
    અચલ uint32_t  ne1 = dst->ne[1];             \
    અચલ uint32_t  ne2 = dst->ne[2];             \
    અચલ uint32_t  ne3 = dst->ne[3];             \
                                                  \
    અચલ uint32_t  nb0 = dst->nb[0];             \
    અચલ uint32_t  nb1 = dst->nb[1];             \
    અચલ uint32_t  nb2 = dst->nb[2];             \
    અચલ uint32_t  nb3 = dst->nb[3];             \

સંરચના sum_rows_context {
    સંરચના htp_ops_context * octx;
    અચલ uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

સ્થિર શૂન્ય sum_rows_thread_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય *data) {
    અચલ સંરચના sum_rows_context * smctx = (અચલ સંરચના sum_rows_context *) data;

    અચલ uint32_t rows_per_thread = smctx->rows_per_thread;
    અચલ uint32_t total_rows      = smctx->total_rows;

    અચલ uint32_t start_row = rows_per_thread * ith;
    અચલ uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    જો (start_row >= end_row) {
        પાછા;
    }

    અચલ size_t   src_stride = smctx->src_stride;
    અચલ size_t   dst_stride = smctx->dst_stride;
    અચલ uint32_t ne00       = smctx->ne00;
    અચલ bool     opt_path   = smctx->opt_path;

    અચલ દશાંશ * restrict src_th = (અચલ દશાંશ *) (smctx->src_data + (start_row * src_stride));
    દશાંશ       * restrict dst_th = (દશાંશ *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    અચલ uint32_t n_rows = end_row - start_row;

    સંરચના htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    માટે (uint32_t ir = 0; ir < n_rows; ir++) {
        અચલ દશાંશ * restrict src_local = (અચલ દશાંશ *) ((અચલ uint8_t *) src_th + ir * src_stride);
        દશાંશ       * restrict dst_local = (દશાંશ *)       ((uint8_t *)       dst_th + ir * dst_stride);

        જો (ir + 1 < n_rows) {
            hex_l2fetch((અચલ uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        જો (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((અચલ uint8_t *) src_local, ne00);
        } નહીંતર {
            *dst_local = hvx_reduce_sum_f32((અચલ uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

પૂર્ણાંક op_sum_rows(સંરચના htp_ops_context * octx) {
    sum_rows_preamble;

    જો (octx->src[0]->type != HTP_TYPE_F32) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    જો (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    અચલ uint32_t src0_nrows      = ne01 * ne02 * ne03;
    અચલ size_t dst_data_row_size = dst->ne[0] * કદ(દશાંશ);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    જો (octx->ctx->mdev.ગણતરી > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, કદ(દશાંશ), (uint32_t) dst_data_row_size, &rows_per_chunk);
        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        row_start = range.start;
        nrows     = range.ગણતરી;
    }

    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    અચલ uint32_t n_threads = octx->n_threads;
    અચલ uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    જો ((0 == hex_is_aligned((શૂન્ય *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    સંરચના sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (અચલ uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    પાછા HTP_STATUS_OK;
}

સંરચના sum_context {
    સંરચના htp_ops_context * octx;
    અચલ દશાંશ            * src_data;
    દશાંશ                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

સ્થિર શૂન્ય sum_thread_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના sum_context * sctx = (સંરચના sum_context *) data;
    અચલ uint32_t start = sctx->elems_per_thread * ith;
    અચલ uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    જો (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        પાછા;
    }

    અચલ uint32_t n = end - start;
    અચલ દશાંશ * src = sctx->src_data + start;

    સંરચના htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((અચલ શૂન્ય *) src, n * કદ(દશાંશ));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((અચલ uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

પૂર્ણાંક op_sum(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * dst  = octx->dst;

    જો (src0->type != HTP_TYPE_F32) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    જો (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    જો (octx->ctx->mdev.ગણતરી > 1 && octx->ctx->mdev.idx > 0) {
        પાછા HTP_STATUS_OK;
    }

    અચલ uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    જો (total_elems == 0) {
        ((દશાંશ *) dst->data)[0] = 0.0f;
        પાછા HTP_STATUS_OK;
    }

    અચલ uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    અચલ uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    અચલ uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    સંરચના sum_context sctx = {
        .octx             = octx,
        .src_data         = (અચલ દશાંશ *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    દશાંશ sum = 0.0f;
    માટે (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((દશાંશ *) dst->data)[0] = sum;

    પાછા HTP_STATUS_OK;
}

સ્થિર અંતઃરેખ શૂન્ય argmax_slice_f32(
    અચલ દશાંશ * restrict src,
    uint32_t n,
    uint32_t offset,
    દશાંશ * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

સંરચના argmax_context {
    સંરચના htp_ops_context * octx;
    અચલ દશાંશ            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    દશાંશ                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

સ્થિર શૂન્ય argmax_thread_single_row(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના argmax_context * actx = (સંરચના argmax_context *) data;
    અચલ uint32_t start = actx->elems_per_thread * ith;
    અચલ uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    જો (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        પાછા;
    }

    અચલ uint32_t n = end - start;
    અચલ દશાંશ * src = actx->src_data + start;

    સંરચના htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((અચલ શૂન્ય *) src, n * કદ(દશાંશ));

    દશાંશ max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

સ્થિર શૂન્ય argmax_thread_multi_row(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના argmax_context * actx = (સંરચના argmax_context *) data;
    અચલ uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    અચલ uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    જો (r0 >= r1) {
        પાછા;
    }

    સંરચના htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    માટે (uint32_t r = r0; r < r1; r++) {
        અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((અચલ શૂન્ય *) src_row, actx->ne00 * કદ(દશાંશ));

        દશાંશ max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

પૂર્ણાંક op_argmax(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * dst  = octx->dst;

    જો (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    જો (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    અચલ uint32_t ne00 = src0->ne[0];
    અચલ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    જો (ne00 == 0 || src0_nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    જો (src0_nrows == 1) {
        જો (octx->ctx->mdev.ગણતરી > 1 && octx->ctx->mdev.idx > 0) {
            પાછા HTP_STATUS_OK;
        }

        જો (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            પાછા HTP_STATUS_OK;
        }

        અચલ uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        અચલ uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        અચલ uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        સંરચના argmax_context actx = {
            .octx             = octx,
            .src_data         = (અચલ દશાંશ *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * કદ(દશાંશ)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) કદ(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        દશાંશ best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        માટે (uint32_t i = 1; i < n_threads; i++) {
            જો (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        પાછા HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    જો (octx->ctx->mdev.ગણતરી > 1) {
        અચલ bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, કદ(int32_t));
        અચલ uint32_t elems_per_chunk = can_split ? 32 : 0;
        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        row_start = range.start;
        nrows     = range.ગણતરી;
    }

    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    અચલ uint32_t n_threads = MIN(octx->n_threads, nrows);
    અચલ uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    સંરચના argmax_context actx = {
        .octx             = octx,
        .src_data         = (અચલ દશાંશ *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * કદ(દશાંશ)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) કદ(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    પાછા HTP_STATUS_OK;
}
