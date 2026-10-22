#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>

#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <math.h>

#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"

#define sum_rows_preamble                         \
    സ്ഥിരം ഘടന htp_tensor *src0 = octx->src[0]; \
    സ്ഥിരം ഘടന htp_tensor *dst  = octx->dst;    \
                                                  \
    സ്ഥിരം uint32_t ne00 = src0->ne[0];            \
    സ്ഥിരം uint32_t ne01 = src0->ne[1];            \
    സ്ഥിരം uint32_t ne02 = src0->ne[2];            \
    സ്ഥിരം uint32_t ne03 = src0->ne[3];            \
                                                  \
    സ്ഥിരം uint32_t nb00 = src0->nb[0];            \
    സ്ഥിരം uint32_t nb01 = src0->nb[1];            \
    സ്ഥിരം uint32_t nb02 = src0->nb[2];            \
    സ്ഥിരം uint32_t nb03 = src0->nb[3];            \
                                                  \
    സ്ഥിരം uint32_t  ne0 = dst->ne[0];             \
    സ്ഥിരം uint32_t  ne1 = dst->ne[1];             \
    സ്ഥിരം uint32_t  ne2 = dst->ne[2];             \
    സ്ഥിരം uint32_t  ne3 = dst->ne[3];             \
                                                  \
    സ്ഥിരം uint32_t  nb0 = dst->nb[0];             \
    സ്ഥിരം uint32_t  nb1 = dst->nb[1];             \
    സ്ഥിരം uint32_t  nb2 = dst->nb[2];             \
    സ്ഥിരം uint32_t  nb3 = dst->nb[3];             \

ഘടന sum_rows_context {
    ഘടന htp_ops_context * octx;
    സ്ഥിരം uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

സ്ഥിതം ശൂന്യം sum_rows_thread_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം *data) {
    സ്ഥിരം ഘടന sum_rows_context * smctx = (സ്ഥിരം ഘടന sum_rows_context *) data;

    സ്ഥിരം uint32_t rows_per_thread = smctx->rows_per_thread;
    സ്ഥിരം uint32_t total_rows      = smctx->total_rows;

    സ്ഥിരം uint32_t start_row = rows_per_thread * ith;
    സ്ഥിരം uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    എങ്കിൽ (start_row >= end_row) {
        തിരികെ;
    }

    സ്ഥിരം size_t   src_stride = smctx->src_stride;
    സ്ഥിരം size_t   dst_stride = smctx->dst_stride;
    സ്ഥിരം uint32_t ne00       = smctx->ne00;
    സ്ഥിരം bool     opt_path   = smctx->opt_path;

    സ്ഥിരം ദശാംശം * restrict src_th = (സ്ഥിരം ദശാംശം *) (smctx->src_data + (start_row * src_stride));
    ദശാംശം       * restrict dst_th = (ദശാംശം *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    സ്ഥിരം uint32_t n_rows = end_row - start_row;

    ഘടന htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    വേണ്ടി (uint32_t ir = 0; ir < n_rows; ir++) {
        സ്ഥിരം ദശാംശം * restrict src_local = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം uint8_t *) src_th + ir * src_stride);
        ദശാംശം       * restrict dst_local = (ദശാംശം *)       ((uint8_t *)       dst_th + ir * dst_stride);

        എങ്കിൽ (ir + 1 < n_rows) {
            hex_l2fetch((സ്ഥിരം uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        എങ്കിൽ (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((സ്ഥിരം uint8_t *) src_local, ne00);
        } അല്ലെങ്കിൽ {
            *dst_local = hvx_reduce_sum_f32((സ്ഥിരം uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

പൂർണ്ണസംഖ്യ op_sum_rows(ഘടന htp_ops_context * octx) {
    sum_rows_preamble;

    എങ്കിൽ (octx->src[0]->type != HTP_TYPE_F32) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    സ്ഥിരം uint32_t src0_nrows      = ne01 * ne02 * ne03;
    സ്ഥിരം size_t dst_data_row_size = dst->ne[0] * വലുപ്പം(ദശാംശം);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, വലുപ്പം(ദശാംശം), (uint32_t) dst_data_row_size, &rows_per_chunk);
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads = octx->n_threads;
    സ്ഥിരം uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    എങ്കിൽ ((0 == hex_is_aligned((ശൂന്യം *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    ഘടന sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (സ്ഥിരം uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    തിരികെ HTP_STATUS_OK;
}

ഘടന sum_context {
    ഘടന htp_ops_context * octx;
    സ്ഥിരം ദശാംശം            * src_data;
    ദശാംശം                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

സ്ഥിതം ശൂന്യം sum_thread_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന sum_context * sctx = (ഘടന sum_context *) data;
    സ്ഥിരം uint32_t start = sctx->elems_per_thread * ith;
    സ്ഥിരം uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    എങ്കിൽ (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        തിരികെ;
    }

    സ്ഥിരം uint32_t n = end - start;
    സ്ഥിരം ദശാംശം * src = sctx->src_data + start;

    ഘടന htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((സ്ഥിരം ശൂന്യം *) src, n * വലുപ്പം(ദശാംശം));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((സ്ഥിരം uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

പൂർണ്ണസംഖ്യ op_sum(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    എങ്കിൽ (src0->type != HTP_TYPE_F32) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1 && octx->ctx->mdev.idx > 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    എങ്കിൽ (total_elems == 0) {
        ((ദശാംശം *) dst->data)[0] = 0.0f;
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    സ്ഥിരം uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    സ്ഥിരം uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    ഘടന sum_context sctx = {
        .octx             = octx,
        .src_data         = (സ്ഥിരം ദശാംശം *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    ദശാംശം sum = 0.0f;
    വേണ്ടി (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((ദശാംശം *) dst->data)[0] = sum;

    തിരികെ HTP_STATUS_OK;
}

സ്ഥിതം വരിയിൽ ശൂന്യം argmax_slice_f32(
    സ്ഥിരം ദശാംശം * restrict src,
    uint32_t n,
    uint32_t offset,
    ദശാംശം * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

ഘടന argmax_context {
    ഘടന htp_ops_context * octx;
    സ്ഥിരം ദശാംശം            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    ദശാംശം                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

സ്ഥിതം ശൂന്യം argmax_thread_single_row(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന argmax_context * actx = (ഘടന argmax_context *) data;
    സ്ഥിരം uint32_t start = actx->elems_per_thread * ith;
    സ്ഥിരം uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    എങ്കിൽ (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        തിരികെ;
    }

    സ്ഥിരം uint32_t n = end - start;
    സ്ഥിരം ദശാംശം * src = actx->src_data + start;

    ഘടന htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((സ്ഥിരം ശൂന്യം *) src, n * വലുപ്പം(ദശാംശം));

    ദശാംശം max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

സ്ഥിതം ശൂന്യം argmax_thread_multi_row(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന argmax_context * actx = (ഘടന argmax_context *) data;
    സ്ഥിരം uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    സ്ഥിരം uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    എങ്കിൽ (r0 >= r1) {
        തിരികെ;
    }

    ഘടന htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    വേണ്ടി (uint32_t r = r0; r < r1; r++) {
        സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((സ്ഥിരം ശൂന്യം *) src_row, actx->ne00 * വലുപ്പം(ദശാംശം));

        ദശാംശം max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

പൂർണ്ണസംഖ്യ op_argmax(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    എങ്കിൽ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    സ്ഥിരം uint32_t ne00 = src0->ne[0];
    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    എങ്കിൽ (ne00 == 0 || src0_nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    എങ്കിൽ (src0_nrows == 1) {
        എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1 && octx->ctx->mdev.idx > 0) {
            തിരികെ HTP_STATUS_OK;
        }

        എങ്കിൽ (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            തിരികെ HTP_STATUS_OK;
        }

        സ്ഥിരം uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        സ്ഥിരം uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        സ്ഥിരം uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        ഘടന argmax_context actx = {
            .octx             = octx,
            .src_data         = (സ്ഥിരം ദശാംശം *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * വലുപ്പം(ദശാംശം)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) വലുപ്പം(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        ദശാംശം best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        വേണ്ടി (uint32_t i = 1; i < n_threads; i++) {
            എങ്കിൽ (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        തിരികെ HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        സ്ഥിരം bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, വലുപ്പം(int32_t));
        സ്ഥിരം uint32_t elems_per_chunk = can_split ? 32 : 0;
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads = MIN(octx->n_threads, nrows);
    സ്ഥിരം uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    ഘടന argmax_context actx = {
        .octx             = octx,
        .src_data         = (സ്ഥിരം ദശാംശം *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * വലുപ്പം(ദശാംശം)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) വലുപ്പം(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    തിരികെ HTP_STATUS_OK;
}
