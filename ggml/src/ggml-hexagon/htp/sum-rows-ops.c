#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#समावेश <string.h>
#समावेश <math.h>

#समावेश "dma-queue.h"
#समावेश "hvx-utils.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "hex-common.h"
#समावेश "hex-profile.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"

#घोषणा योग_rows_preamble                         \
    स्थिर काष्ठा htp_tensor *src0 = octx->src[0]; \
    स्थिर काष्ठा htp_tensor *dst  = octx->dst;    \
                                                  \
    स्थिर uint32_t ne00 = src0->ne[0];            \
    स्थिर uint32_t ne01 = src0->ne[1];            \
    स्थिर uint32_t ne02 = src0->ne[2];            \
    स्थिर uint32_t ne03 = src0->ne[3];            \
                                                  \
    स्थिर uint32_t nb00 = src0->nb[0];            \
    स्थिर uint32_t nb01 = src0->nb[1];            \
    स्थिर uint32_t nb02 = src0->nb[2];            \
    स्थिर uint32_t nb03 = src0->nb[3];            \
                                                  \
    स्थिर uint32_t  ne0 = dst->ne[0];             \
    स्थिर uint32_t  ne1 = dst->ne[1];             \
    स्थिर uint32_t  ne2 = dst->ne[2];             \
    स्थिर uint32_t  ne3 = dst->ne[3];             \
                                                  \
    स्थिर uint32_t  nb0 = dst->nb[0];             \
    स्थिर uint32_t  nb1 = dst->nb[1];             \
    स्थिर uint32_t  nb2 = dst->nb[2];             \
    स्थिर uint32_t  nb3 = dst->nb[3];             \

काष्ठा योग_rows_context {
    काष्ठा htp_ops_context * octx;
    स्थिर uint8_t * src_तथ्य;
    uint8_t       * dst_तथ्य;
    uint32_t        ne00;
    माप_प्रकार          src_stride;
    माप_प्रकार          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        योग_rows;
    bool            opt_path;
};

अटल व्योम योग_rows_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम *तथ्य) {
    स्थिर काष्ठा योग_rows_context * smctx = (स्थिर काष्ठा योग_rows_context *) तथ्य;

    स्थिर uint32_t rows_per_thread = smctx->rows_per_thread;
    स्थिर uint32_t योग_rows      = smctx->योग_rows;

    स्थिर uint32_t start_row = rows_per_thread * ith;
    स्थिर uint32_t end_row   = MIN(start_row + rows_per_thread, योग_rows);

    अगर (start_row >= end_row) {
        वापस;
    }

    स्थिर माप_प्रकार   src_stride = smctx->src_stride;
    स्थिर माप_प्रकार   dst_stride = smctx->dst_stride;
    स्थिर uint32_t ne00       = smctx->ne00;
    स्थिर bool     opt_path   = smctx->opt_path;

    स्थिर भग्न * restrict src_th = (स्थिर भग्न *) (smctx->src_तथ्य + (start_row * src_stride));
    भग्न       * restrict dst_th = (भग्न *)       (smctx->dst_तथ्य + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    स्थिर uint32_t n_rows = end_row - start_row;

    काष्ठा htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    क्रम (uint32_t ir = 0; ir < n_rows; ir++) {
        स्थिर भग्न * restrict src_local = (स्थिर भग्न *) ((स्थिर uint8_t *) src_th + ir * src_stride);
        भग्न       * restrict dst_local = (भग्न *)       ((uint8_t *)       dst_th + ir * dst_stride);

        अगर (ir + 1 < n_rows) {
            hex_l2fetch((स्थिर uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        अगर (opt_path) {
            *dst_local = hvx_reduce_योग_f32_a((स्थिर uint8_t *) src_local, ne00);
        } अन्यथा {
            *dst_local = hvx_reduce_योग_f32((स्थिर uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

पूर्णांक op_योग_rows(काष्ठा htp_ops_context * octx) {
    योग_rows_preamble;

    अगर (octx->src[0]->type != HTP_TYPE_F32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर uint32_t src0_nrows      = ne01 * ne02 * ne03;
    स्थिर माप_प्रकार dst_तथ्य_row_आकार = dst->ne[0] * माप(भग्न);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, माप(भग्न), (uint32_t) dst_तथ्य_row_आकार, &rows_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = octx->n_threads;
    स्थिर uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    अगर ((0 == hex_is_aligned((व्योम *) src0->तथ्य, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    काष्ठा योग_rows_context smctx = {
        .octx            = octx,
        .src_तथ्य        = (स्थिर uint8_t *) src0->तथ्य + row_start * nb01,
        .dst_तथ्य        = (uint8_t *) dst->तथ्य + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .योग_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, योग_rows_thread_f32, &smctx, n_threads);

    वापस HTP_STATUS_OK;
}

काष्ठा योग_context {
    काष्ठा htp_ops_context * octx;
    स्थिर भग्न            * src_तथ्य;
    भग्न                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 योग_elems;
    uint32_t                 elems_per_thread;
};

अटल व्योम योग_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा योग_context * sctx = (काष्ठा योग_context *) तथ्य;
    स्थिर uint32_t start = sctx->elems_per_thread * ith;
    स्थिर uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->योग_elems);

    अगर (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        वापस;
    }

    स्थिर uint32_t n = end - start;
    स्थिर भग्न * src = sctx->src_तथ्य + start;

    काष्ठा htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((स्थिर व्योम *) src, n * माप(भग्न));

    sctx->partial_sums[ith] = hvx_reduce_योग_f32((स्थिर uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

पूर्णांक op_योग(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    अगर (src0->type != HTP_TYPE_F32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (octx->ctx->mdev.गिनती > 1 && octx->ctx->mdev.idx > 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t योग_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    अगर (योग_elems == 0) {
        ((भग्न *) dst->तथ्य)[0] = 0.0f;
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = (योग_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    स्थिर uint32_t raw_chunk = (योग_elems + n_threads - 1) / n_threads;
    स्थिर uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    काष्ठा योग_context sctx = {
        .octx             = octx,
        .src_तथ्य         = (स्थिर भग्न *) src0->तथ्य,
        .योग_elems      = योग_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, योग_thread_f32, &sctx, n_threads);

    भग्न योग_2 = 0.0f;
    क्रम (uint32_t i = 0; i < n_threads; i++) {
        योग_2 += sctx.partial_sums[i];
    }
    ((भग्न *) dst->तथ्य)[0] = योग_2;

    वापस HTP_STATUS_OK;
}

अटल अंतरभूत व्योम argmax_slice_f32(
    स्थिर भग्न * restrict src,
    uint32_t n,
    uint32_t offset,
    भग्न * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

काष्ठा argmax_context {
    काष्ठा htp_ops_context * octx;
    स्थिर भग्न            * src_तथ्य;
    int32_t                * dst_तथ्य;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    भग्न                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

अटल व्योम argmax_thread_single_row(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा argmax_context * actx = (काष्ठा argmax_context *) तथ्य;
    स्थिर uint32_t start = actx->elems_per_thread * ith;
    स्थिर uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    अगर (start >= end) {
        actx->partial_max[ith] = -अनन्त;
        actx->partial_idx[ith] = 0;
        वापस;
    }

    स्थिर uint32_t n = end - start;
    स्थिर भग्न * src = actx->src_तथ्य + start;

    काष्ठा htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((स्थिर व्योम *) src, n * माप(भग्न));

    भग्न max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

अटल व्योम argmax_thread_multi_row(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा argmax_context * actx = (काष्ठा argmax_context *) तथ्य;
    स्थिर uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    स्थिर uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    अगर (r0 >= r1) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    क्रम (uint32_t r = r0; r < r1; r++) {
        स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर uint8_t *) actx->src_तथ्य + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_तथ्य + r * actx->dst_stride);

        hex_l2fetch_block((स्थिर व्योम *) src_row, actx->ne00 * माप(भग्न));

        भग्न max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

पूर्णांक op_argmax(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    अगर (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर uint32_t ne00 = src0->ne[0];
    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    अगर (ne00 == 0 || src0_nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (src0_nrows == 1) {
        अगर (octx->ctx->mdev.गिनती > 1 && octx->ctx->mdev.idx > 0) {
            वापस HTP_STATUS_OK;
        }

        अगर (ne00 == 1) {
            ((int32_t *) dst->तथ्य)[0] = 0;
            वापस HTP_STATUS_OK;
        }

        स्थिर uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        स्थिर uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        स्थिर uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        काष्ठा argmax_context actx = {
            .octx             = octx,
            .src_तथ्य         = (स्थिर भग्न *) src0->तथ्य,
            .dst_तथ्य         = (int32_t *) dst->तथ्य,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * माप(भग्न)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) माप(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        भग्न best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        क्रम (uint32_t i = 1; i < n_threads; i++) {
            अगर (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->तथ्य)[0] = best_idx;
        वापस HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर bool can_split = htp_tensor_mdev_तथ्य_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, माप(int32_t));
        स्थिर uint32_t elems_per_chunk = can_split ? 32 : 0;
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = MIN(octx->n_threads, nrows);
    स्थिर uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    काष्ठा argmax_context actx = {
        .octx             = octx,
        .src_तथ्य         = (स्थिर भग्न *) src0->तथ्य,
        .dst_तथ्य         = (int32_t *) dst->तथ्य,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * माप(भग्न)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) माप(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    वापस HTP_STATUS_OK;
}
