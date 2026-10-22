#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <string.h>
#अन्तर्भाव <math.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"

#define sum_rows_preamble                         \
    ध्रुव संरचना htp_tensor *src0 = octx->src[0]; \
    ध्रुव संरचना htp_tensor *dst  = octx->dst;    \
                                                  \
    ध्रुव uint32_t ne00 = src0->ne[0];            \
    ध्रुव uint32_t ne01 = src0->ne[1];            \
    ध्रुव uint32_t ne02 = src0->ne[2];            \
    ध्रुव uint32_t ne03 = src0->ne[3];            \
                                                  \
    ध्रुव uint32_t nb00 = src0->nb[0];            \
    ध्रुव uint32_t nb01 = src0->nb[1];            \
    ध्रुव uint32_t nb02 = src0->nb[2];            \
    ध्रुव uint32_t nb03 = src0->nb[3];            \
                                                  \
    ध्रुव uint32_t  ne0 = dst->ne[0];             \
    ध्रुव uint32_t  ne1 = dst->ne[1];             \
    ध्रुव uint32_t  ne2 = dst->ne[2];             \
    ध्रुव uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ध्रुव uint32_t  nb0 = dst->nb[0];             \
    ध्रुव uint32_t  nb1 = dst->nb[1];             \
    ध्रुव uint32_t  nb2 = dst->nb[2];             \
    ध्रुव uint32_t  nb3 = dst->nb[3];             \

संरचना sum_rows_context {
    संरचना htp_ops_context * octx;
    ध्रुव uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

स्थिर शून्य sum_rows_thread_f32(unsigned पूर्णाङ्क nth, unsigned पूर्णाङ्क ith, शून्य *data) {
    ध्रुव संरचना sum_rows_context * smctx = (ध्रुव संरचना sum_rows_context *) data;

    ध्रुव uint32_t rows_per_thread = smctx->rows_per_thread;
    ध्रुव uint32_t total_rows      = smctx->total_rows;

    ध्रुव uint32_t start_row = rows_per_thread * ith;
    ध्रुव uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    सचे (start_row >= end_row) {
        पटिनिवत्त;
    }

    ध्रुव size_t   src_stride = smctx->src_stride;
    ध्रुव size_t   dst_stride = smctx->dst_stride;
    ध्रुव uint32_t ne00       = smctx->ne00;
    ध्रुव bool     opt_path   = smctx->opt_path;

    ध्रुव भिन्न * restrict src_th = (ध्रुव भिन्न *) (smctx->src_data + (start_row * src_stride));
    भिन्न       * restrict dst_th = (भिन्न *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    ध्रुव uint32_t n_rows = end_row - start_row;

    संरचना htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    अत्थाय (uint32_t ir = 0; ir < n_rows; ir++) {
        ध्रुव भिन्न * restrict src_local = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src_th + ir * src_stride);
        भिन्न       * restrict dst_local = (भिन्न *)       ((uint8_t *)       dst_th + ir * dst_stride);

        सचे (ir + 1 < n_rows) {
            hex_l2fetch((ध्रुव uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        सचे (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((ध्रुव uint8_t *) src_local, ne00);
        } अञ्ञथा {
            *dst_local = hvx_reduce_sum_f32((ध्रुव uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

पूर्णाङ्क op_sum_rows(संरचना htp_ops_context * octx) {
    sum_rows_preamble;

    सचे (octx->src[0]->type != HTP_TYPE_F32) {
        पटिनिवत्त HTP_STATUS_NO_SUPPORT;
    }

    सचे (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        पटिनिवत्त HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ध्रुव size_t dst_data_row_size = dst->ne[0] * परिमाण(भिन्न);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    सचे (octx->ctx->mdev.गणनम् > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, परिमाण(भिन्न), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    सचे (nrows == 0) {
        पटिनिवत्त HTP_STATUS_OK;
    }

    ध्रुव uint32_t n_threads = octx->n_threads;
    ध्रुव uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    सचे ((0 == hex_is_aligned((शून्य *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    संरचना sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (ध्रुव uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    पटिनिवत्त HTP_STATUS_OK;
}

संरचना sum_context {
    संरचना htp_ops_context * octx;
    ध्रुव भिन्न            * src_data;
    भिन्न                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

स्थिर शून्य sum_thread_f32(unsigned पूर्णाङ्क nth, unsigned पूर्णाङ्क ith, शून्य * data) {
    संरचना sum_context * sctx = (संरचना sum_context *) data;
    ध्रुव uint32_t start = sctx->elems_per_thread * ith;
    ध्रुव uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    सचे (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        पटिनिवत्त;
    }

    ध्रुव uint32_t n = end - start;
    ध्रुव भिन्न * src = sctx->src_data + start;

    संरचना htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ध्रुव शून्य *) src, n * परिमाण(भिन्न));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((ध्रुव uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

पूर्णाङ्क op_sum(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    सचे (src0->type != HTP_TYPE_F32) {
        पटिनिवत्त HTP_STATUS_NO_SUPPORT;
    }

    सचे (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        पटिनिवत्त HTP_STATUS_NO_SUPPORT;
    }

    सचे (octx->ctx->mdev.गणनम् > 1 && octx->ctx->mdev.idx > 0) {
        पटिनिवत्त HTP_STATUS_OK;
    }

    ध्रुव uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    सचे (total_elems == 0) {
        ((भिन्न *) dst->data)[0] = 0.0f;
        पटिनिवत्त HTP_STATUS_OK;
    }

    ध्रुव uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    ध्रुव uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    ध्रुव uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    संरचना sum_context sctx = {
        .octx             = octx,
        .src_data         = (ध्रुव भिन्न *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    भिन्न sum = 0.0f;
    अत्थाय (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((भिन्न *) dst->data)[0] = sum;

    पटिनिवत्त HTP_STATUS_OK;
}

स्थिर अन्तःस्थ शून्य argmax_slice_f32(
    ध्रुव भिन्न * restrict src,
    uint32_t n,
    uint32_t offset,
    भिन्न * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

संरचना argmax_context {
    संरचना htp_ops_context * octx;
    ध्रुव भिन्न            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    भिन्न                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

स्थिर शून्य argmax_thread_single_row(unsigned पूर्णाङ्क nth, unsigned पूर्णाङ्क ith, शून्य * data) {
    संरचना argmax_context * actx = (संरचना argmax_context *) data;
    ध्रुव uint32_t start = actx->elems_per_thread * ith;
    ध्रुव uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    सचे (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        पटिनिवत्त;
    }

    ध्रुव uint32_t n = end - start;
    ध्रुव भिन्न * src = actx->src_data + start;

    संरचना htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ध्रुव शून्य *) src, n * परिमाण(भिन्न));

    भिन्न max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

स्थिर शून्य argmax_thread_multi_row(unsigned पूर्णाङ्क nth, unsigned पूर्णाङ्क ith, शून्य * data) {
    संरचना argmax_context * actx = (संरचना argmax_context *) data;
    ध्रुव uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    ध्रुव uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    सचे (r0 >= r1) {
        पटिनिवत्त;
    }

    संरचना htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    अत्थाय (uint32_t r = r0; r < r1; r++) {
        ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((ध्रुव शून्य *) src_row, actx->ne00 * परिमाण(भिन्न));

        भिन्न max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

पूर्णाङ्क op_argmax(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    सचे (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        पटिनिवत्त HTP_STATUS_NO_SUPPORT;
    }

    सचे (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        पटिनिवत्त HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव uint32_t ne00 = src0->ne[0];
    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    सचे (ne00 == 0 || src0_nrows == 0) {
        पटिनिवत्त HTP_STATUS_OK;
    }

    सचे (src0_nrows == 1) {
        सचे (octx->ctx->mdev.गणनम् > 1 && octx->ctx->mdev.idx > 0) {
            पटिनिवत्त HTP_STATUS_OK;
        }

        सचे (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            पटिनिवत्त HTP_STATUS_OK;
        }

        ध्रुव uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        ध्रुव uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        ध्रुव uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        संरचना argmax_context actx = {
            .octx             = octx,
            .src_data         = (ध्रुव भिन्न *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * परिमाण(भिन्न)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) परिमाण(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        भिन्न best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        अत्थाय (uint32_t i = 1; i < n_threads; i++) {
            सचे (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        पटिनिवत्त HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    सचे (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, परिमाण(int32_t));
        ध्रुव uint32_t elems_per_chunk = can_split ? 32 : 0;
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    सचे (nrows == 0) {
        पटिनिवत्त HTP_STATUS_OK;
    }

    ध्रुव uint32_t n_threads = MIN(octx->n_threads, nrows);
    ध्रुव uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    संरचना argmax_context actx = {
        .octx             = octx,
        .src_data         = (ध्रुव भिन्न *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * परिमाण(भिन्न)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) परिमाण(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    पटिनिवत्त HTP_STATUS_OK;
}
