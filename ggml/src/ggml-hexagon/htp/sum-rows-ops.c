#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <string.h>
#अन्तर्भाव <math.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"

#परिभाषय योग_पङ्क्तिकाः_preamble                         \
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

संरचना योग_पङ्क्तिकाः_context {
    संरचना htp_ops_context * octx;
    ध्रुव uint8_t * src_दत्त;
    uint8_t       * dst_दत्त;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        पङ्क्तिकाः_प्रत्येकम्_thread;
    uint32_t        योग_पङ्क्तिकाः;
    bool            opt_मार्गः;
};

स्थिर शून्य योग_पङ्क्तिकाः_thread_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य *दत्त) {
    ध्रुव संरचना योग_पङ्क्तिकाः_context * smctx = (ध्रुव संरचना योग_पङ्क्तिकाः_context *) दत्त;

    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_thread = smctx->पङ्क्तिकाः_प्रत्येकम्_thread;
    ध्रुव uint32_t योग_पङ्क्तिकाः      = smctx->योग_पङ्क्तिकाः;

    ध्रुव uint32_t start_row = पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_row   = MIN(start_row + पङ्क्तिकाः_प्रत्येकम्_thread, योग_पङ्क्तिकाः);

    यदि (start_row >= अन्तः_row) {
        प्रत्यागम;
    }

    ध्रुव size_t   src_stride = smctx->src_stride;
    ध्रुव size_t   dst_stride = smctx->dst_stride;
    ध्रुव uint32_t ne00       = smctx->ne00;
    ध्रुव bool     opt_मार्गः   = smctx->opt_मार्गः;

    ध्रुव भिन्न * सीमित src_th = (ध्रुव भिन्न *) (smctx->src_दत्त + (start_row * src_stride));
    भिन्न       * सीमित dst_th = (भिन्न *)       (smctx->dst_दत्त + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    ध्रुव uint32_t n_पङ्क्तिकाः = अन्तः_row - start_row;

    संरचना htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    प्रति (uint32_t ir = 0; ir < n_पङ्क्तिकाः; ir++) {
        ध्रुव भिन्न * सीमित src_local = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src_th + ir * src_stride);
        भिन्न       * सीमित dst_local = (भिन्न *)       ((uint8_t *)       dst_th + ir * dst_stride);

        यदि (ir + 1 < n_पङ्क्तिकाः) {
            hex_l2fetch((ध्रुव uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        यदि (opt_मार्गः) {
            *dst_local = hvx_reduce_योग_f32_a((ध्रुव uint8_t *) src_local, ne00);
        } अन्यथा {
            *dst_local = hvx_reduce_योग_f32((ध्रुव uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

पूर्णाङ्क op_योग_पङ्क्तिकाः(संरचना htp_ops_context * octx) {
    योग_पङ्क्तिकाः_preamble;

    यदि (octx->src[0]->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ध्रुव size_t dst_दत्त_row_आकार = dst->ne[0] * परिमाण(भिन्न);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, परिमाण(भिन्न), (uint32_t) dst_दत्त_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, पङ्क्तिकाः_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;
    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_thread = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div);

    bool opt_मार्गः = false;
    यदि ((0 == hex_is_aligned((शून्य *) src0->दत्त, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_मार्गः = true;
    }

    संरचना योग_पङ्क्तिकाः_context smctx = {
        .octx            = octx,
        .src_दत्त        = (ध्रुव uint8_t *) src0->दत्त + row_start * nb01,
        .dst_दत्त        = (uint8_t *) dst->दत्त + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .पङ्क्तिकाः_प्रत्येकम्_thread = पङ्क्तिकाः_प्रत्येकम्_thread,
        .योग_पङ्क्तिकाः      = nrows,
        .opt_मार्गः        = opt_मार्गः,
    };

    work_queue_सततम्(octx->ctx->work_queue, योग_पङ्क्तिकाः_thread_f32, &smctx, n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}

संरचना योग_context {
    संरचना htp_ops_context * octx;
    ध्रुव भिन्न            * src_दत्त;
    भिन्न                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 योग_elems;
    uint32_t                 elems_प्रत्येकम्_thread;
};

स्थिर शून्य योग_thread_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना योग_context * sctx = (संरचना योग_context *) दत्त;
    ध्रुव uint32_t start = sctx->elems_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_2   = MIN(start + sctx->elems_प्रत्येकम्_thread, sctx->योग_elems);

    यदि (start >= अन्तः_2) {
        sctx->partial_sums[ith] = 0.0f;
        प्रत्यागम;
    }

    ध्रुव uint32_t n = अन्तः_2 - start;
    ध्रुव भिन्न * src = sctx->src_दत्त + start;

    संरचना htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_खण्डम्((ध्रुव शून्य *) src, n * परिमाण(भिन्न));

    sctx->partial_sums[ith] = hvx_reduce_योग_f32((ध्रुव uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

पूर्णाङ्क op_योग(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    यदि (src0->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (octx->ctx->mdev.गणनम् > 1 && octx->ctx->mdev.idx > 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t योग_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    यदि (योग_elems == 0) {
        ((भिन्न *) dst->दत्त)[0] = 0.0f;
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = (योग_elems >= 1024) ? MIN(octx->n_तन्तवः, HTP_MAX_NTHREADS) : 1;
    ध्रुव uint32_t raw_chunk = (योग_elems + n_तन्तवः - 1) / n_तन्तवः;
    ध्रुव uint32_t elems_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(raw_chunk, 32);

    संरचना योग_context sctx = {
        .octx             = octx,
        .src_दत्त         = (ध्रुव भिन्न *) src0->दत्त,
        .योग_elems      = योग_elems,
        .elems_प्रत्येकम्_thread = elems_प्रत्येकम्_thread,
    };

    work_queue_सततम्(octx->ctx->work_queue, योग_thread_f32, &sctx, n_तन्तवः);

    भिन्न योग_2 = 0.0f;
    प्रति (uint32_t i = 0; i < n_तन्तवः; i++) {
        योग_2 += sctx.partial_sums[i];
    }
    ((भिन्न *) dst->दत्त)[0] = योग_2;

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर अन्तःस्थ शून्य argmax_खण्डिका_f32(
    ध्रुव भिन्न * सीमित src,
    uint32_t n,
    uint32_t विस्थापनम्,
    भिन्न * बहिर्गमनम्_val,
    int32_t * बहिर्गमनम्_idx
) {
    hvx_argmax_f32(src, n, विस्थापनम्, बहिर्गमनम्_val, बहिर्गमनम्_idx);
}

संरचना argmax_context {
    संरचना htp_ops_context * octx;
    ध्रुव भिन्न            * src_दत्त;
    int32_t                * dst_दत्त;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 पङ्क्तिकाः_प्रत्येकम्_thread;
    uint32_t                 elems_प्रत्येकम्_thread;

    भिन्न                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

स्थिर शून्य argmax_thread_single_row(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना argmax_context * actx = (संरचना argmax_context *) दत्त;
    ध्रुव uint32_t start = actx->elems_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_2   = MIN(start + actx->elems_प्रत्येकम्_thread, actx->ne00);

    यदि (start >= अन्तः_2) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        प्रत्यागम;
    }

    ध्रुव uint32_t n = अन्तः_2 - start;
    ध्रुव भिन्न * src = actx->src_दत्त + start;

    संरचना htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_खण्डम्((ध्रुव शून्य *) src, n * परिमाण(भिन्न));

    भिन्न max_val;
    int32_t max_idx;
    argmax_खण्डिका_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

स्थिर शून्य argmax_thread_multi_row(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना argmax_context * actx = (संरचना argmax_context *) दत्त;
    ध्रुव uint32_t r0 = actx->row_start + actx->पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t r1 = MIN(r0 + actx->पङ्क्तिकाः_प्रत्येकम्_thread, actx->row_start + actx->nrows);

    यदि (r0 >= r1) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    प्रति (uint32_t r = r0; r < r1; r++) {
        ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) actx->src_दत्त + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_दत्त + r * actx->dst_stride);

        hex_l2fetch_खण्डम्((ध्रुव शून्य *) src_row, actx->ne00 * परिमाण(भिन्न));

        भिन्न max_val;
        int32_t max_idx;
        argmax_खण्डिका_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

पूर्णाङ्क op_argmax(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    यदि (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव uint32_t ne00 = src0->ne[0];
    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    यदि (ne00 == 0 || src0_nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    यदि (src0_nrows == 1) {
        यदि (octx->ctx->mdev.गणनम् > 1 && octx->ctx->mdev.idx > 0) {
            प्रत्यागम HTP_स्थितिः_OK;
        }

        यदि (ne00 == 1) {
            ((int32_t *) dst->दत्त)[0] = 0;
            प्रत्यागम HTP_स्थितिः_OK;
        }

        ध्रुव uint32_t n_तन्तवः = (ne00 >= 1024) ? MIN(octx->n_तन्तवः, HTP_MAX_NTHREADS) : 1;
        ध्रुव uint32_t raw_chunk = (ne00 + n_तन्तवः - 1) / n_तन्तवः;
        ध्रुव uint32_t elems_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(raw_chunk, 32);

        संरचना argmax_context actx = {
            .octx             = octx,
            .src_दत्त         = (ध्रुव भिन्न *) src0->दत्त,
            .dst_दत्त         = (int32_t *) dst->दत्त,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * परिमाण(भिन्न)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) परिमाण(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .पङ्क्तिकाः_प्रत्येकम्_thread  = 1,
            .elems_प्रत्येकम्_thread = elems_प्रत्येकम्_thread,
        };

        work_queue_सततम्(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_तन्तवः);

        भिन्न best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        प्रति (uint32_t i = 1; i < n_तन्तवः; i++) {
            यदि (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->दत्त)[0] = best_idx;
        प्रत्यागम HTP_स्थितिः_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव bool can_विभज = htp_tensor_mdev_दत्त_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, परिमाण(int32_t));
        ध्रुव uint32_t elems_प्रत्येकम्_chunk = can_विभज ? 32 : 0;
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = MIN(octx->n_तन्तवः, nrows);
    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_thread = (nrows + n_तन्तवः - 1) / n_तन्तवः;

    संरचना argmax_context actx = {
        .octx             = octx,
        .src_दत्त         = (ध्रुव भिन्न *) src0->दत्त,
        .dst_दत्त         = (int32_t *) dst->दत्त,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * परिमाण(भिन्न)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) परिमाण(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .पङ्क्तिकाः_प्रत्येकम्_thread  = पङ्क्तिकाः_प्रत्येकम्_thread,
        .elems_प्रत्येकम्_thread = 0,
    };

    work_queue_सततम्(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_तन्तवः);
    प्रत्यागम HTP_स्थितिः_OK;
}
