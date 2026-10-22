#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"

संरचना htp_roll_context {
    संरचना htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    संरचना fastdiv_मानहरू div_ne1;
    संरचना fastdiv_मानहरू div_ne2_ne1;
};

स्थिर अन्तःस्थ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    यदि (i < 0) {
        फर्काऊ (uint32_t) (i + (int32_t) ne);
    }
    यदि ((uint32_t) i >= ne) {
        फर्काऊ (uint32_t) i - ne;
    }
    फर्काऊ (uint32_t) i;
}

#define htp_roll_preamble                             \
    अचर संरचना htp_tensor * src0 = octx->src[0];    \
    अचर संरचना htp_tensor * dst  = octx->dst;       \
                                                      \
    अचर uint32_t ne0 = dst->ne[0];                  \
    अचर uint32_t ne1 = dst->ne[1];                  \
    अचर uint32_t ne2 = dst->ne[2];                  \
    अचर uint32_t ne3 = dst->ne[3];                  \
                                                      \
    अचर uint32_t nb01 = src0->nb[1];                \
    अचर uint32_t nb02 = src0->nb[2];                \
    अचर uint32_t nb03 = src0->nb[3];                \
                                                      \
    अचर uint32_t nb1 = dst->nb[1];                  \
    अचर uint32_t nb2 = dst->nb[2];                  \
    अचर uint32_t nb3 = dst->nb[3];                  \
                                                      \
    अचर int32_t s0 = octx->op_params[0];            \
    अचर int32_t s1 = octx->op_params[1];            \
    अचर int32_t s2 = octx->op_params[2];            \
    अचर int32_t s3 = octx->op_params[3];            \
                                                      \
    अचर uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    अचर uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

स्थिर अन्तःस्थ शून्य roll_dma_थप(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    यदि (bytes == 0 || nrows == 0) {
        फर्काऊ;
    }

    यदि (!dma_queue_थप(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_थप(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

स्थिर अन्तःस्थ शून्य roll_dma_थप_rows(dma_queue *               q,
                                      अचर संरचना htp_tensor * dst,
                                      अचर संरचना htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0) {
    अचर dma_addr_t dst_base = dst->data + (माप_प्रकार) dst_row * row_size;
    अचर dma_addr_t src_base = src0->data + (माप_प्रकार) src_row * row_size;
    अचर uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_थप(q, dst_base, src_base + (माप_प्रकार) i0_src0 * परिमाण(भिन्न),
                  row_size, row_size, n0 * परिमाण(भिन्न), nrows);
    roll_dma_थप(q, dst_base + (माप_प्रकार) n0 * परिमाण(भिन्न), src_base,
                  row_size, row_size, i0_src0 * परिमाण(भिन्न), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
स्थिर अन्तःस्थ शून्य roll_dma_थप_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_थप(q, dst_row, src_row + (माप_प्रकार) i0_src0 * परिमाण(भिन्न),
                  dst_stride, src_stride, n0 * परिमाण(भिन्न), nrows);
    roll_dma_थप(q, dst_row + (माप_प्रकार) n0 * परिमाण(भिन्न), src_row,
                  dst_stride, src_stride, i0_src0 * परिमाण(भिन्न), nrows);
}

स्थिर पूर्णाङ्क roll_dma_f32_contiguous(संरचना htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    अचर uint32_t row_size = ne0 * परिमाण(भिन्न);

    यदि (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_थप_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0);
        dma_queue_flush(q);
        फर्काऊ HTP_STATUS_OK;
    }

    यदि (s1 == 0) {
        अचर uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        लागि (uint32_t i3 = 0; i3 < ne3; i3++) {
            अचर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            अचर uint32_t dst_row0 = i3 * ne2 * ne1;
            अचर uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            अचर uint32_t n2_first = ne2 - i2_src0;

            roll_dma_थप_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0);
            roll_dma_थप_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0);
        }

        dma_queue_flush(q);
        फर्काऊ HTP_STATUS_OK;
    }

    अचर uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    अचर uint32_t n1_first = ne1 - i1_src0;

    लागि (uint32_t i3 = 0; i3 < ne3; i3++) {
        अचर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        लागि (uint32_t i2 = 0; i2 < ne2; i2++) {
            अचर uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            अचर uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            अचर uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_थप_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0);
            roll_dma_थप_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0);
        }
    }

    dma_queue_flush(q);
    फर्काऊ HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
स्थिर पूर्णाङ्क roll_dma_f32_strided(संरचना htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    अचर uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    अचर uint32_t n1_first = ne1 - i1_src0;

    लागि (uint32_t i3 = 0; i3 < ne3; i3++) {
        अचर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        लागि (uint32_t i2 = 0; i2 < ne2; i2++) {
            अचर uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            अचर dma_addr_t dst_row0 = dst->data  + (माप_प्रकार) i2  * nb2  + (माप_प्रकार) i3  * nb3;
            अचर dma_addr_t src_row0 = src0->data + (माप_प्रकार) i02 * nb02 + (माप_प्रकार) i03 * nb03;

            roll_dma_थप_range(q, dst_row0, src_row0 + (माप_प्रकार) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_थप_range(q, dst_row0 + (माप_प्रकार) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    फर्काऊ HTP_STATUS_OK;
}

स्थिर शून्य roll_thread_f32(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_roll_context * rctx = (संरचना htp_roll_context *) data;
    संरचना htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    अचर uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    अचर uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    यदि (row_start >= row_end) {
        फर्काऊ;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    लागि (uint32_t row = row_start; row < row_end; row++) {
        अचर uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        अचर uint32_t rem = row - i3 * ne2 * ne1;
        अचर uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        अचर uint32_t i1  = rem - i2 * ne1;

        अचर uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        अचर uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        अचर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        अचर uint8_t * src_row = (अचर uint8_t *) (uintptr_t) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->data + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * परिमाण(भिन्न), n0 * परिमाण(भिन्न), ne0 * परिमाण(भिन्न), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * परिमाण(भिन्न), n0, परिमाण(भिन्न));

        यदि (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * परिमाण(भिन्न), ne0 * परिमाण(भिन्न), 1);
            hvx_copy_uu(dst_row + n0 * परिमाण(भिन्न), src_row, i0_src0, परिमाण(भिन्न));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(माथिल्लो_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

पूर्णाङ्क execute_op_roll_f32(संरचना htp_ops_context * octx) {
    htp_roll_preamble;

    यदि (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    यदि (src0->nb[0] != परिमाण(भिन्न) || dst->nb[0] != परिमाण(भिन्न)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    यदि (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        फर्काऊ HTP_STATUS_INVAL_PARAMS;
    }

    अचर uint32_t total_rows = ne1 * ne2 * ne3;
    अचर माप_प्रकार dst_row_size = ne0 * परिमाण(भिन्न);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    यदि (octx->ctx->mdev.गन्ती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, परिमाण(भिन्न), (uint32_t) dst_row_size, &rows_per_chunk);
        अचर संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गन्ती, &octx->ctx->mdev.गन्ती_div);
        row_start = range.start;
        nrows     = range.गन्ती;
    }

    यदि (nrows == 0) {
        फर्काऊ HTP_STATUS_OK;
    }

    यदि (octx->ctx->mdev.गन्ती <= 1) {
        यदि (htp_tensor_is_contiguous(src0, परिमाण(भिन्न)) && htp_tensor_is_contiguous(dst, परिमाण(भिन्न))) {
            फर्काऊ roll_dma_f32_contiguous(octx);
        }
        फर्काऊ roll_dma_f32_strided(octx);
    }

    यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    अचर uint32_t n_threads = octx->n_threads;
    संरचना htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_मानहरू(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_मानहरू(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    फर्काऊ HTP_STATUS_OK;
}

पूर्णाङ्क op_roll(संरचना htp_ops_context * octx) {
    विकल्प (octx->src[0]->type) {
        पक्ष HTP_TYPE_F32:
            फर्काऊ execute_op_roll_f32(octx);

        स्वभाव:
            फर्काऊ HTP_STATUS_NO_SUPPORT;
    }
}
