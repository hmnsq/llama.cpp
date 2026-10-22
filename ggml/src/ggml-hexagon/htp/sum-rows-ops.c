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
    स्थिरांक संरचना htp_tensor *src0 = octx->src[0]; \
    स्थिरांक संरचना htp_tensor *dst  = octx->dst;    \
                                                  \
    स्थिरांक uint32_t ne00 = src0->ne[0];            \
    स्थिरांक uint32_t ne01 = src0->ne[1];            \
    स्थिरांक uint32_t ne02 = src0->ne[2];            \
    स्थिरांक uint32_t ne03 = src0->ne[3];            \
                                                  \
    स्थिरांक uint32_t nb00 = src0->nb[0];            \
    स्थिरांक uint32_t nb01 = src0->nb[1];            \
    स्थिरांक uint32_t nb02 = src0->nb[2];            \
    स्थिरांक uint32_t nb03 = src0->nb[3];            \
                                                  \
    स्थिरांक uint32_t  ne0 = dst->ne[0];             \
    स्थिरांक uint32_t  ne1 = dst->ne[1];             \
    स्थिरांक uint32_t  ne2 = dst->ne[2];             \
    स्थिरांक uint32_t  ne3 = dst->ne[3];             \
                                                  \
    स्थिरांक uint32_t  nb0 = dst->nb[0];             \
    स्थिरांक uint32_t  nb1 = dst->nb[1];             \
    स्थिरांक uint32_t  nb2 = dst->nb[2];             \
    स्थिरांक uint32_t  nb3 = dst->nb[3];             \

संरचना sum_rows_context {
    संरचना htp_ops_context * octx;
    स्थिरांक uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    माप_प्रकार          src_stride;
    माप_प्रकार          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

स्थिर शून्य sum_rows_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य *data) {
    स्थिरांक संरचना sum_rows_context * smctx = (स्थिरांक संरचना sum_rows_context *) data;

    स्थिरांक uint32_t rows_per_thread = smctx->rows_per_thread;
    स्थिरांक uint32_t total_rows      = smctx->total_rows;

    स्थिरांक uint32_t start_row = rows_per_thread * ith;
    स्थिरांक uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    जर (start_row >= end_row) {
        परत;
    }

    स्थिरांक माप_प्रकार   src_stride = smctx->src_stride;
    स्थिरांक माप_प्रकार   dst_stride = smctx->dst_stride;
    स्थिरांक uint32_t ne00       = smctx->ne00;
    स्थिरांक bool     opt_path   = smctx->opt_path;

    स्थिरांक दशांश * restrict src_th = (स्थिरांक दशांश *) (smctx->src_data + (start_row * src_stride));
    दशांश       * restrict dst_th = (दशांश *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    स्थिरांक uint32_t n_rows = end_row - start_row;

    संरचना htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    साठी (uint32_t ir = 0; ir < n_rows; ir++) {
        स्थिरांक दशांश * restrict src_local = (स्थिरांक दशांश *) ((स्थिरांक uint8_t *) src_th + ir * src_stride);
        दशांश       * restrict dst_local = (दशांश *)       ((uint8_t *)       dst_th + ir * dst_stride);

        जर (ir + 1 < n_rows) {
            hex_l2fetch((स्थिरांक uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        जर (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((स्थिरांक uint8_t *) src_local, ne00);
        } अन्यथा {
            *dst_local = hvx_reduce_sum_f32((स्थिरांक uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

पूर्णांक op_sum_rows(संरचना htp_ops_context * octx) {
    sum_rows_preamble;

    जर (octx->src[0]->type != HTP_TYPE_F32) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    स्थिरांक uint32_t src0_nrows      = ne01 * ne02 * ne03;
    स्थिरांक माप_प्रकार dst_data_row_size = dst->ne[0] * आकार(दशांश);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    जर (octx->ctx->mdev.मोजणी > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, आकार(दशांश), (uint32_t) dst_data_row_size, &rows_per_chunk);
        स्थिरांक संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.मोजणी, &octx->ctx->mdev.मोजणी_div);
        row_start = range.start;
        nrows     = range.मोजणी;
    }

    जर (nrows == 0) {
        परत HTP_STATUS_OK;
    }

    स्थिरांक uint32_t n_threads = octx->n_threads;
    स्थिरांक uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    जर ((0 == hex_is_aligned((शून्य *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    संरचना sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (स्थिरांक uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    परत HTP_STATUS_OK;
}

संरचना sum_context {
    संरचना htp_ops_context * octx;
    स्थिरांक दशांश            * src_data;
    दशांश                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

स्थिर शून्य sum_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {
    संरचना sum_context * sctx = (संरचना sum_context *) data;
    स्थिरांक uint32_t start = sctx->elems_per_thread * ith;
    स्थिरांक uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    जर (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        परत;
    }

    स्थिरांक uint32_t n = end - start;
    स्थिरांक दशांश * src = sctx->src_data + start;

    संरचना htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((स्थिरांक शून्य *) src, n * आकार(दशांश));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((स्थिरांक uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

पूर्णांक op_sum(संरचना htp_ops_context * octx) {
    स्थिरांक संरचना htp_tensor * src0 = octx->src[0];
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;

    जर (src0->type != HTP_TYPE_F32) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (octx->ctx->mdev.मोजणी > 1 && octx->ctx->mdev.idx > 0) {
        परत HTP_STATUS_OK;
    }

    स्थिरांक uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    जर (total_elems == 0) {
        ((दशांश *) dst->data)[0] = 0.0f;
        परत HTP_STATUS_OK;
    }

    स्थिरांक uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    स्थिरांक uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    स्थिरांक uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    संरचना sum_context sctx = {
        .octx             = octx,
        .src_data         = (स्थिरांक दशांश *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    दशांश sum = 0.0f;
    साठी (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((दशांश *) dst->data)[0] = sum;

    परत HTP_STATUS_OK;
}

स्थिर ओळीत शून्य argmax_slice_f32(
    स्थिरांक दशांश * restrict src,
    uint32_t n,
    uint32_t offset,
    दशांश * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

संरचना argmax_context {
    संरचना htp_ops_context * octx;
    स्थिरांक दशांश            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    दशांश                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

स्थिर शून्य argmax_thread_single_row(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {
    संरचना argmax_context * actx = (संरचना argmax_context *) data;
    स्थिरांक uint32_t start = actx->elems_per_thread * ith;
    स्थिरांक uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    जर (start >= end) {
        actx->partial_max[ith] = -अनन्त;
        actx->partial_idx[ith] = 0;
        परत;
    }

    स्थिरांक uint32_t n = end - start;
    स्थिरांक दशांश * src = actx->src_data + start;

    संरचना htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((स्थिरांक शून्य *) src, n * आकार(दशांश));

    दशांश max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

स्थिर शून्य argmax_thread_multi_row(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {
    संरचना argmax_context * actx = (संरचना argmax_context *) data;
    स्थिरांक uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    स्थिरांक uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    जर (r0 >= r1) {
        परत;
    }

    संरचना htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    साठी (uint32_t r = r0; r < r1; r++) {
        स्थिरांक दशांश * src_row = (स्थिरांक दशांश *) ((स्थिरांक uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((स्थिरांक शून्य *) src_row, actx->ne00 * आकार(दशांश));

        दशांश max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

पूर्णांक op_argmax(संरचना htp_ops_context * octx) {
    स्थिरांक संरचना htp_tensor * src0 = octx->src[0];
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;

    जर (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    स्थिरांक uint32_t ne00 = src0->ne[0];
    स्थिरांक uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    जर (ne00 == 0 || src0_nrows == 0) {
        परत HTP_STATUS_OK;
    }

    जर (src0_nrows == 1) {
        जर (octx->ctx->mdev.मोजणी > 1 && octx->ctx->mdev.idx > 0) {
            परत HTP_STATUS_OK;
        }

        जर (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            परत HTP_STATUS_OK;
        }

        स्थिरांक uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        स्थिरांक uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        स्थिरांक uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        संरचना argmax_context actx = {
            .octx             = octx,
            .src_data         = (स्थिरांक दशांश *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * आकार(दशांश)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) आकार(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        दशांश best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        साठी (uint32_t i = 1; i < n_threads; i++) {
            जर (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        परत HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    जर (octx->ctx->mdev.मोजणी > 1) {
        स्थिरांक bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, आकार(int32_t));
        स्थिरांक uint32_t elems_per_chunk = can_split ? 32 : 0;
        स्थिरांक संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.मोजणी, &octx->ctx->mdev.मोजणी_div);
        row_start = range.start;
        nrows     = range.मोजणी;
    }

    जर (nrows == 0) {
        परत HTP_STATUS_OK;
    }

    स्थिरांक uint32_t n_threads = MIN(octx->n_threads, nrows);
    स्थिरांक uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    संरचना argmax_context actx = {
        .octx             = octx,
        .src_data         = (स्थिरांक दशांश *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * आकार(दशांश)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) आकार(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    परत HTP_STATUS_OK;
}
