#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <math.h>

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"

#define sum_rows_preamble                         \
    ଅଚଳ ସଂରଚନା htp_tensor *src0 = octx->src[0]; \
    ଅଚଳ ସଂରଚନା htp_tensor *dst  = octx->dst;    \
                                                  \
    ଅଚଳ uint32_t ne00 = src0->ne[0];            \
    ଅଚଳ uint32_t ne01 = src0->ne[1];            \
    ଅଚଳ uint32_t ne02 = src0->ne[2];            \
    ଅଚଳ uint32_t ne03 = src0->ne[3];            \
                                                  \
    ଅଚଳ uint32_t nb00 = src0->nb[0];            \
    ଅଚଳ uint32_t nb01 = src0->nb[1];            \
    ଅଚଳ uint32_t nb02 = src0->nb[2];            \
    ଅଚଳ uint32_t nb03 = src0->nb[3];            \
                                                  \
    ଅଚଳ uint32_t  ne0 = dst->ne[0];             \
    ଅଚଳ uint32_t  ne1 = dst->ne[1];             \
    ଅଚଳ uint32_t  ne2 = dst->ne[2];             \
    ଅଚଳ uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ଅଚଳ uint32_t  nb0 = dst->nb[0];             \
    ଅଚଳ uint32_t  nb1 = dst->nb[1];             \
    ଅଚଳ uint32_t  nb2 = dst->nb[2];             \
    ଅଚଳ uint32_t  nb3 = dst->nb[3];             \

ସଂରଚନା sum_rows_context {
    ସଂରଚନା htp_ops_context * octx;
    ଅଚଳ uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

ସ୍ଥିର ଶୂନ୍ୟ sum_rows_thread_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ *data) {
    ଅଚଳ ସଂରଚନା sum_rows_context * smctx = (ଅଚଳ ସଂରଚନା sum_rows_context *) data;

    ଅଚଳ uint32_t rows_per_thread = smctx->rows_per_thread;
    ଅଚଳ uint32_t total_rows      = smctx->total_rows;

    ଅଚଳ uint32_t start_row = rows_per_thread * ith;
    ଅଚଳ uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    ଯଦି (start_row >= end_row) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ size_t   src_stride = smctx->src_stride;
    ଅଚଳ size_t   dst_stride = smctx->dst_stride;
    ଅଚଳ uint32_t ne00       = smctx->ne00;
    ଅଚଳ bool     opt_path   = smctx->opt_path;

    ଅଚଳ ଦଶମିକ * restrict src_th = (ଅଚଳ ଦଶମିକ *) (smctx->src_data + (start_row * src_stride));
    ଦଶମିକ       * restrict dst_th = (ଦଶମିକ *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    ଅଚଳ uint32_t n_rows = end_row - start_row;

    ସଂରଚନା htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    ପାଇଁ (uint32_t ir = 0; ir < n_rows; ir++) {
        ଅଚଳ ଦଶମିକ * restrict src_local = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) src_th + ir * src_stride);
        ଦଶମିକ       * restrict dst_local = (ଦଶମିକ *)       ((uint8_t *)       dst_th + ir * dst_stride);

        ଯଦି (ir + 1 < n_rows) {
            hex_l2fetch((ଅଚଳ uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        ଯଦି (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((ଅଚଳ uint8_t *) src_local, ne00);
        } ଅନ୍ୟଥା {
            *dst_local = hvx_reduce_sum_f32((ଅଚଳ uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_sum_rows(ସଂରଚନା htp_ops_context * octx) {
    sum_rows_preamble;

    ଯଦି (octx->src[0]->type != HTP_TYPE_F32) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ଅଚଳ size_t dst_data_row_size = dst->ne[0] * ଆକାର(ଦଶମିକ);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ଆକାର(ଦଶମିକ), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;
    ଅଚଳ uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    ଯଦି ((0 == hex_is_aligned((ଶୂନ୍ୟ *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    ସଂରଚନା sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (ଅଚଳ uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସଂରଚନା sum_context {
    ସଂରଚନା htp_ops_context * octx;
    ଅଚଳ ଦଶମିକ            * src_data;
    ଦଶମିକ                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

ସ୍ଥିର ଶୂନ୍ୟ sum_thread_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା sum_context * sctx = (ସଂରଚନା sum_context *) data;
    ଅଚଳ uint32_t start = sctx->elems_per_thread * ith;
    ଅଚଳ uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    ଯଦି (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ uint32_t n = end - start;
    ଅଚଳ ଦଶମିକ * src = sctx->src_data + start;

    ସଂରଚନା htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ଅଚଳ ଶୂନ୍ୟ *) src, n * ଆକାର(ଦଶମିକ));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((ଅଚଳ uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_sum(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଯଦି (src0->type != HTP_TYPE_F32) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1 && octx->ctx->mdev.idx > 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ଯଦି (total_elems == 0) {
        ((ଦଶମିକ *) dst->data)[0] = 0.0f;
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    ଅଚଳ uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    ଅଚଳ uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    ସଂରଚନା sum_context sctx = {
        .octx             = octx,
        .src_data         = (ଅଚଳ ଦଶମିକ *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    ଦଶମିକ sum = 0.0f;
    ପାଇଁ (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((ଦଶମିକ *) dst->data)[0] = sum;

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ argmax_slice_f32(
    ଅଚଳ ଦଶମିକ * restrict src,
    uint32_t n,
    uint32_t offset,
    ଦଶମିକ * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

ସଂରଚନା argmax_context {
    ସଂରଚନା htp_ops_context * octx;
    ଅଚଳ ଦଶମିକ            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    ଦଶମିକ                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ସ୍ଥିର ଶୂନ୍ୟ argmax_thread_single_row(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା argmax_context * actx = (ସଂରଚନା argmax_context *) data;
    ଅଚଳ uint32_t start = actx->elems_per_thread * ith;
    ଅଚଳ uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    ଯଦି (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ uint32_t n = end - start;
    ଅଚଳ ଦଶମିକ * src = actx->src_data + start;

    ସଂରଚନା htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ଅଚଳ ଶୂନ୍ୟ *) src, n * ଆକାର(ଦଶମିକ));

    ଦଶମିକ max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

ସ୍ଥିର ଶୂନ୍ୟ argmax_thread_multi_row(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା argmax_context * actx = (ସଂରଚନା argmax_context *) data;
    ଅଚଳ uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    ଅଚଳ uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    ଯଦି (r0 >= r1) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    ପାଇଁ (uint32_t r = r0; r < r1; r++) {
        ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((ଅଚଳ ଶୂନ୍ୟ *) src_row, actx->ne00 * ଆକାର(ଦଶମିକ));

        ଦଶମିକ max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_argmax(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଯଦି (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t ne00 = src0->ne[0];
    ଅଚଳ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    ଯଦି (ne00 == 0 || src0_nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଯଦି (src0_nrows == 1) {
        ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1 && octx->ctx->mdev.idx > 0) {
            ଫେରନ୍ତୁ HTP_STATUS_OK;
        }

        ଯଦି (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            ଫେରନ୍ତୁ HTP_STATUS_OK;
        }

        ଅଚଳ uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        ଅଚଳ uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        ଅଚଳ uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        ସଂରଚନା argmax_context actx = {
            .octx             = octx,
            .src_data         = (ଅଚଳ ଦଶମିକ *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ଆକାର(ଦଶମିକ)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ଆକାର(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        ଦଶମିକ best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        ପାଇଁ (uint32_t i = 1; i < n_threads; i++) {
            ଯଦି (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        ଅଚଳ bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, ଆକାର(int32_t));
        ଅଚଳ uint32_t elems_per_chunk = can_split ? 32 : 0;
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = MIN(octx->n_threads, nrows);
    ଅଚଳ uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    ସଂରଚନା argmax_context actx = {
        .octx             = octx,
        .src_data         = (ଅଚଳ ଦଶମିକ *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ଆକାର(ଦଶମିକ)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ଆକାର(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    ଫେରନ୍ତୁ HTP_STATUS_OK;
}
