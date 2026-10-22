#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#অন্তর্ভাব <string.h>
#অন্তর্ভাব <math.h>

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hvx-utils.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"

#define যোগ_rows_preamble                         \
    ধ্রুবক কাঠামো htp_tensor *src0 = octx->src[0]; \
    ধ্রুবক কাঠামো htp_tensor *dst  = octx->dst;    \
                                                  \
    ধ্রুবক uint32_t ne00 = src0->ne[0];            \
    ধ্রুবক uint32_t ne01 = src0->ne[1];            \
    ধ্রুবক uint32_t ne02 = src0->ne[2];            \
    ধ্রুবক uint32_t ne03 = src0->ne[3];            \
                                                  \
    ধ্রুবক uint32_t nb00 = src0->nb[0];            \
    ধ্রুবক uint32_t nb01 = src0->nb[1];            \
    ধ্রুবক uint32_t nb02 = src0->nb[2];            \
    ধ্রুবক uint32_t nb03 = src0->nb[3];            \
                                                  \
    ধ্রুবক uint32_t  ne0 = dst->ne[0];             \
    ধ্রুবক uint32_t  ne1 = dst->ne[1];             \
    ধ্রুবক uint32_t  ne2 = dst->ne[2];             \
    ধ্রুবক uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ধ্রুবক uint32_t  nb0 = dst->nb[0];             \
    ধ্রুবক uint32_t  nb1 = dst->nb[1];             \
    ধ্রুবক uint32_t  nb2 = dst->nb[2];             \
    ধ্রুবক uint32_t  nb3 = dst->nb[3];             \

কাঠামো যোগ_rows_context {
    কাঠামো htp_ops_context * octx;
    ধ্রুবক uint8_t * src_তথ্য;
    uint8_t       * dst_তথ্য;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        যোগ_rows;
    bool            opt_path;
};

স্থির শূন্য যোগ_rows_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য *তথ্য) {
    ধ্রুবক কাঠামো যোগ_rows_context * smctx = (ধ্রুবক কাঠামো যোগ_rows_context *) তথ্য;

    ধ্রুবক uint32_t rows_per_thread = smctx->rows_per_thread;
    ধ্রুবক uint32_t যোগ_rows      = smctx->যোগ_rows;

    ধ্রুবক uint32_t start_row = rows_per_thread * ith;
    ধ্রুবক uint32_t end_row   = MIN(start_row + rows_per_thread, যোগ_rows);

    যদি (start_row >= end_row) {
        ফেরত;
    }

    ধ্রুবক size_t   src_stride = smctx->src_stride;
    ধ্রুবক size_t   dst_stride = smctx->dst_stride;
    ধ্রুবক uint32_t ne00       = smctx->ne00;
    ধ্রুবক bool     opt_path   = smctx->opt_path;

    ধ্রুবক দশমিক * restrict src_th = (ধ্রুবক দশমিক *) (smctx->src_তথ্য + (start_row * src_stride));
    দশমিক       * restrict dst_th = (দশমিক *)       (smctx->dst_তথ্য + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    ধ্রুবক uint32_t n_rows = end_row - start_row;

    কাঠামো htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    জন্য (uint32_t ir = 0; ir < n_rows; ir++) {
        ধ্রুবক দশমিক * restrict src_local = (ধ্রুবক দশমিক *) ((ধ্রুবক uint8_t *) src_th + ir * src_stride);
        দশমিক       * restrict dst_local = (দশমিক *)       ((uint8_t *)       dst_th + ir * dst_stride);

        যদি (ir + 1 < n_rows) {
            hex_l2fetch((ধ্রুবক uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        যদি (opt_path) {
            *dst_local = hvx_reduce_যোগ_f32_a((ধ্রুবক uint8_t *) src_local, ne00);
        } অন্যথা {
            *dst_local = hvx_reduce_যোগ_f32((ধ্রুবক uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

পূর্ণসংখ্যা op_যোগ_rows(কাঠামো htp_ops_context * octx) {
    যোগ_rows_preamble;

    যদি (octx->src[0]->type != HTP_TYPE_F32) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ধ্রুবক size_t dst_তথ্য_row_আকার = dst->ne[0] * আকার(দশমিক);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, আকার(দশমিক), (uint32_t) dst_তথ্য_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;
    ধ্রুবক uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    যদি ((0 == hex_is_aligned((শূন্য *) src0->তথ্য, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    কাঠামো যোগ_rows_context smctx = {
        .octx            = octx,
        .src_তথ্য        = (ধ্রুবক uint8_t *) src0->তথ্য + row_start * nb01,
        .dst_তথ্য        = (uint8_t *) dst->তথ্য + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .যোগ_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, যোগ_rows_thread_f32, &smctx, n_threads);

    ফেরত HTP_STATUS_OK;
}

কাঠামো যোগ_context {
    কাঠামো htp_ops_context * octx;
    ধ্রুবক দশমিক            * src_তথ্য;
    দশমিক                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 যোগ_elems;
    uint32_t                 elems_per_thread;
};

স্থির শূন্য যোগ_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো যোগ_context * sctx = (কাঠামো যোগ_context *) তথ্য;
    ধ্রুবক uint32_t start = sctx->elems_per_thread * ith;
    ধ্রুবক uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->যোগ_elems);

    যদি (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        ফেরত;
    }

    ধ্রুবক uint32_t n = end - start;
    ধ্রুবক দশমিক * src = sctx->src_তথ্য + start;

    কাঠামো htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ধ্রুবক শূন্য *) src, n * আকার(দশমিক));

    sctx->partial_sums[ith] = hvx_reduce_যোগ_f32((ধ্রুবক uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

পূর্ণসংখ্যা op_যোগ(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    যদি (src0->type != HTP_TYPE_F32) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (octx->ctx->mdev.গুনতি > 1 && octx->ctx->mdev.idx > 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t যোগ_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    যদি (যোগ_elems == 0) {
        ((দশমিক *) dst->তথ্য)[0] = 0.0f;
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = (যোগ_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    ধ্রুবক uint32_t raw_chunk = (যোগ_elems + n_threads - 1) / n_threads;
    ধ্রুবক uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    কাঠামো যোগ_context sctx = {
        .octx             = octx,
        .src_তথ্য         = (ধ্রুবক দশমিক *) src0->তথ্য,
        .যোগ_elems      = যোগ_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, যোগ_thread_f32, &sctx, n_threads);

    দশমিক যোগ_2 = 0.0f;
    জন্য (uint32_t i = 0; i < n_threads; i++) {
        যোগ_2 += sctx.partial_sums[i];
    }
    ((দশমিক *) dst->তথ্য)[0] = যোগ_2;

    ফেরত HTP_STATUS_OK;
}

স্থির অন্তঃরেখ শূন্য argmax_slice_f32(
    ধ্রুবক দশমিক * restrict src,
    uint32_t n,
    uint32_t offset,
    দশমিক * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

কাঠামো argmax_context {
    কাঠামো htp_ops_context * octx;
    ধ্রুবক দশমিক            * src_তথ্য;
    int32_t                * dst_তথ্য;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    দশমিক                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

স্থির শূন্য argmax_thread_single_row(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো argmax_context * actx = (কাঠামো argmax_context *) তথ্য;
    ধ্রুবক uint32_t start = actx->elems_per_thread * ith;
    ধ্রুবক uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    যদি (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        ফেরত;
    }

    ধ্রুবক uint32_t n = end - start;
    ধ্রুবক দশমিক * src = actx->src_তথ্য + start;

    কাঠামো htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((ধ্রুবক শূন্য *) src, n * আকার(দশমিক));

    দশমিক max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

স্থির শূন্য argmax_thread_multi_row(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো argmax_context * actx = (কাঠামো argmax_context *) তথ্য;
    ধ্রুবক uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    ধ্রুবক uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    যদি (r0 >= r1) {
        ফেরত;
    }

    কাঠামো htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    জন্য (uint32_t r = r0; r < r1; r++) {
        ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক uint8_t *) actx->src_তথ্য + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_তথ্য + r * actx->dst_stride);

        hex_l2fetch_block((ধ্রুবক শূন্য *) src_row, actx->ne00 * আকার(দশমিক));

        দশমিক max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

পূর্ণসংখ্যা op_argmax(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    যদি (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t ne00 = src0->ne[0];
    ধ্রুবক uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    যদি (ne00 == 0 || src0_nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    যদি (src0_nrows == 1) {
        যদি (octx->ctx->mdev.গুনতি > 1 && octx->ctx->mdev.idx > 0) {
            ফেরত HTP_STATUS_OK;
        }

        যদি (ne00 == 1) {
            ((int32_t *) dst->তথ্য)[0] = 0;
            ফেরত HTP_STATUS_OK;
        }

        ধ্রুবক uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        ধ্রুবক uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        ধ্রুবক uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        কাঠামো argmax_context actx = {
            .octx             = octx,
            .src_তথ্য         = (ধ্রুবক দশমিক *) src0->তথ্য,
            .dst_তথ্য         = (int32_t *) dst->তথ্য,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * আকার(দশমিক)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) আকার(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        দশমিক best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        জন্য (uint32_t i = 1; i < n_threads; i++) {
            যদি (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->তথ্য)[0] = best_idx;
        ফেরত HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুবক bool can_split = htp_tensor_mdev_তথ্য_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, আকার(int32_t));
        ধ্রুবক uint32_t elems_per_chunk = can_split ? 32 : 0;
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = MIN(octx->n_threads, nrows);
    ধ্রুবক uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    কাঠামো argmax_context actx = {
        .octx             = octx,
        .src_তথ্য         = (ধ্রুবক দশমিক *) src0->তথ্য,
        .dst_তথ্য         = (int32_t *) dst->তথ্য,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * আকার(দশমিক)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) আকার(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    ফেরত HTP_STATUS_OK;
}
