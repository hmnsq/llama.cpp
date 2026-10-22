#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <math.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

#define sum_rows_preamble                         \
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor *src0 = octx->src[0]; \
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor *dst  = octx->dst;    \
                                                  \
    ಧ್ರುವ uint32_t ne00 = src0->ne[0];            \
    ಧ್ರುವ uint32_t ne01 = src0->ne[1];            \
    ಧ್ರುವ uint32_t ne02 = src0->ne[2];            \
    ಧ್ರುವ uint32_t ne03 = src0->ne[3];            \
                                                  \
    ಧ್ರುವ uint32_t nb00 = src0->nb[0];            \
    ಧ್ರುವ uint32_t nb01 = src0->nb[1];            \
    ಧ್ರುವ uint32_t nb02 = src0->nb[2];            \
    ಧ್ರುವ uint32_t nb03 = src0->nb[3];            \
                                                  \
    ಧ್ರುವ uint32_t  ne0 = dst->ne[0];             \
    ಧ್ರುವ uint32_t  ne1 = dst->ne[1];             \
    ಧ್ರುವ uint32_t  ne2 = dst->ne[2];             \
    ಧ್ರುವ uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ಧ್ರುವ uint32_t  nb0 = dst->nb[0];             \
    ಧ್ರುವ uint32_t  nb1 = dst->nb[1];             \
    ಧ್ರುವ uint32_t  nb2 = dst->nb[2];             \
    ಧ್ರುವ uint32_t  nb3 = dst->nb[3];             \

ಸಂರಚನಾ sum_rows_context {
    ಸಂರಚನಾ htp_ops_context * octx;
    ಧ್ರುವ uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

ಸ್ಥಿರ ಶೂನ್ಯ sum_rows_thread_f32(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ *data) {
    ಧ್ರುವ ಸಂರಚನಾ sum_rows_context * smctx = (ಧ್ರುವ ಸಂರಚನಾ sum_rows_context *) data;

    ಧ್ರುವ uint32_t rows_per_thread = smctx->rows_per_thread;
    ಧ್ರುವ uint32_t total_rows      = smctx->total_rows;

    ಧ್ರುವ uint32_t start_row = rows_per_thread * ith;
    ಧ್ರುವ uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    ಯದಿ (start_row >= end_row) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ size_t   src_stride = smctx->src_stride;
    ಧ್ರುವ size_t   dst_stride = smctx->dst_stride;
    ಧ್ರುವ uint32_t ne00       = smctx->ne00;
    ಧ್ರುವ bool     opt_path   = smctx->opt_path;

    ಧ್ರುವ ಭಿನ್ನ * restrict src_th = (ಧ್ರುವ ಭಿನ್ನ *) (smctx->src_data + (start_row * src_stride));
    ಭಿನ್ನ       * restrict dst_th = (ಭಿನ್ನ *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    ಧ್ರುವ uint32_t n_rows = end_row - start_row;

    ಸಂರಚನಾ htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    ಪ್ರತಿ (uint32_t ir = 0; ir < n_rows; ir++) {
        ಧ್ರುವ ಭಿನ್ನ * restrict src_local = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ uint8_t *) src_th + ir * src_stride);
        ಭಿನ್ನ       * restrict dst_local = (ಭಿನ್ನ *)       ((uint8_t *)       dst_th + ir * dst_stride);

        ಯದಿ (ir + 1 < n_rows) {
            hex_l2fetch((ಧ್ರುವ uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        ಯದಿ (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((ಧ್ರುವ uint8_t *) src_local, ne00);
        } ಅನ್ಯಥಾ {
            *dst_local = hvx_reduce_sum_f32((ಧ್ರುವ uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

ಪೂರ್ಣಾಙ್ಕ op_sum_rows(ಸಂರಚನಾ htp_ops_context * octx) {
    sum_rows_preamble;

    ಯದಿ (octx->src[0]->type != HTP_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಧ್ರುವ uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ಧ್ರುವ size_t dst_data_row_size = dst->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ಪರಿಮಾಣ(ಭಿನ್ನ), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_start = range.start;
        nrows     = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t n_threads = octx->n_threads;
    ಧ್ರುವ uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    ಯದಿ ((0 == hex_is_aligned((ಶೂನ್ಯ *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    ಸಂರಚನಾ sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (ಧ್ರುವ uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಸಂರಚನಾ sum_context {
    ಸಂರಚನಾ htp_ops_context * octx;
    ಧ್ರುವ ಭಿನ್ನ            * src_data;
    ಭಿನ್ನ                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ sum_thread_f32(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ sum_context * sctx = (ಸಂರಚನಾ sum_context *) data;
    ಧ್ರುವ uint32_t start = sctx->elems_per_thread * ith;
    ಧ್ರುವ uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    ಯದಿ (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint32_t n = end - start;
    ಧ್ರುವ ಭಿನ್ನ * src = sctx->src_data + start;

    ಸಂರಚನಾ htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ಧ್ರುವ ಶೂನ್ಯ *) src, n * ಪರಿಮಾಣ(ಭಿನ್ನ));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((ಧ್ರುವ uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

ಪೂರ್ಣಾಙ್ಕ op_sum(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಯದಿ (src0->type != HTP_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1 && octx->ctx->mdev.idx > 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ಯದಿ (total_elems == 0) {
        ((ಭಿನ್ನ *) dst->data)[0] = 0.0f;
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    ಧ್ರುವ uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    ಧ್ರುವ uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    ಸಂರಚನಾ sum_context sctx = {
        .octx             = octx,
        .src_data         = (ಧ್ರುವ ಭಿನ್ನ *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    ಭಿನ್ನ sum = 0.0f;
    ಪ್ರತಿ (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((ಭಿನ್ನ *) dst->data)[0] = sum;

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ argmax_slice_f32(
    ಧ್ರುವ ಭಿನ್ನ * restrict src,
    uint32_t n,
    uint32_t offset,
    ಭಿನ್ನ * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

ಸಂರಚನಾ argmax_context {
    ಸಂರಚನಾ htp_ops_context * octx;
    ಧ್ರುವ ಭಿನ್ನ            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    ಭಿನ್ನ                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ಸ್ಥಿರ ಶೂನ್ಯ argmax_thread_single_row(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ argmax_context * actx = (ಸಂರಚನಾ argmax_context *) data;
    ಧ್ರುವ uint32_t start = actx->elems_per_thread * ith;
    ಧ್ರುವ uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    ಯದಿ (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint32_t n = end - start;
    ಧ್ರುವ ಭಿನ್ನ * src = actx->src_data + start;

    ಸಂರಚನಾ htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ಧ್ರುವ ಶೂನ್ಯ *) src, n * ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಭಿನ್ನ max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

ಸ್ಥಿರ ಶೂನ್ಯ argmax_thread_multi_row(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ argmax_context * actx = (ಸಂರಚನಾ argmax_context *) data;
    ಧ್ರುವ uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    ಧ್ರುವ uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    ಯದಿ (r0 >= r1) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಸಂರಚನಾ htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    ಪ್ರತಿ (uint32_t r = r0; r < r1; r++) {
        ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((ಧ್ರುವ ಶೂನ್ಯ *) src_row, actx->ne00 * ಪರಿಮಾಣ(ಭಿನ್ನ));

        ಭಿನ್ನ max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

ಪೂರ್ಣಾಙ್ಕ op_argmax(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಯದಿ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಧ್ರುವ uint32_t ne00 = src0->ne[0];
    ಧ್ರುವ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    ಯದಿ (ne00 == 0 || src0_nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಯದಿ (src0_nrows == 1) {
        ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1 && octx->ctx->mdev.idx > 0) {
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
        }

        ಯದಿ (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
        }

        ಧ್ರುವ uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        ಧ್ರುವ uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        ಧ್ರುವ uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        ಸಂರಚನಾ argmax_context actx = {
            .octx             = octx,
            .src_data         = (ಧ್ರುವ ಭಿನ್ನ *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ಪರಿಮಾಣ(ಭಿನ್ನ)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ಪರಿಮಾಣ(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        ಭಿನ್ನ best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        ಪ್ರತಿ (uint32_t i = 1; i < n_threads; i++) {
            ಯದಿ (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        ಧ್ರುವ bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, ಪರಿಮಾಣ(int32_t));
        ಧ್ರುವ uint32_t elems_per_chunk = can_split ? 32 : 0;
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_start = range.start;
        nrows     = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t n_threads = MIN(octx->n_threads, nrows);
    ಧ್ರುವ uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    ಸಂರಚನಾ argmax_context actx = {
        .octx             = octx,
        .src_data         = (ಧ್ರುವ ಭಿನ್ನ *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ಪರಿಮಾಣ(ಭಿನ್ನ)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ಪರಿಮಾಣ(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}
