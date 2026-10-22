#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#அந்தர்பாவ <string.h>

#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"

கட்டமைப்பு htp_roll_context {
    கட்டமைப்பு htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne1;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne2_ne1;
};

நிலையான உள்வரி uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    எனில் (i < 0) {
        திரும்பு (uint32_t) (i + (int32_t) ne);
    }
    எனில் ((uint32_t) i >= ne) {
        திரும்பு (uint32_t) i - ne;
    }
    திரும்பு (uint32_t) i;
}

#define htp_roll_preamble                             \
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];    \
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;       \
                                                      \
    மாறிலி uint32_t ne0 = dst->ne[0];                  \
    மாறிலி uint32_t ne1 = dst->ne[1];                  \
    மாறிலி uint32_t ne2 = dst->ne[2];                  \
    மாறிலி uint32_t ne3 = dst->ne[3];                  \
                                                      \
    மாறிலி uint32_t nb01 = src0->nb[1];                \
    மாறிலி uint32_t nb02 = src0->nb[2];                \
    மாறிலி uint32_t nb03 = src0->nb[3];                \
                                                      \
    மாறிலி uint32_t nb1 = dst->nb[1];                  \
    மாறிலி uint32_t nb2 = dst->nb[2];                  \
    மாறிலி uint32_t nb3 = dst->nb[3];                  \
                                                      \
    மாறிலி int32_t s0 = octx->op_params[0];            \
    மாறிலி int32_t s1 = octx->op_params[1];            \
    மாறிலி int32_t s2 = octx->op_params[2];            \
    மாறிலி int32_t s3 = octx->op_params[3];            \
                                                      \
    மாறிலி uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    மாறிலி uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

நிலையான உள்வரி வெறுமை roll_dma_தள்ளு(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    எனில் (bytes == 0 || nrows == 0) {
        திரும்பு;
    }

    எனில் (!dma_queue_தள்ளு(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_தள்ளு(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

நிலையான உள்வரி வெறுமை roll_dma_தள்ளு_rows(dma_queue *               q,
                                      மாறிலி கட்டமைப்பு htp_tensor * dst,
                                      மாறிலி கட்டமைப்பு htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0) {
    மாறிலி dma_addr_t dst_base = dst->data + (size_t) dst_row * row_size;
    மாறிலி dma_addr_t src_base = src0->data + (size_t) src_row * row_size;
    மாறிலி uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_தள்ளு(q, dst_base, src_base + (size_t) i0_src0 * அளவு(தசம),
                  row_size, row_size, n0 * அளவு(தசம), nrows);
    roll_dma_தள்ளு(q, dst_base + (size_t) n0 * அளவு(தசம), src_base,
                  row_size, row_size, i0_src0 * அளவு(தசம), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
நிலையான உள்வரி வெறுமை roll_dma_தள்ளு_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_தள்ளு(q, dst_row, src_row + (size_t) i0_src0 * அளவு(தசம),
                  dst_stride, src_stride, n0 * அளவு(தசம), nrows);
    roll_dma_தள்ளு(q, dst_row + (size_t) n0 * அளவு(தசம), src_row,
                  dst_stride, src_stride, i0_src0 * அளவு(தசம), nrows);
}

நிலையான முழு roll_dma_f32_contiguous(கட்டமைப்பு htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    மாறிலி uint32_t row_size = ne0 * அளவு(தசம);

    எனில் (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_தள்ளு_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0);
        dma_queue_flush(q);
        திரும்பு HTP_STATUS_OK;
    }

    எனில் (s1 == 0) {
        மாறிலி uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        க்கு (uint32_t i3 = 0; i3 < ne3; i3++) {
            மாறிலி uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            மாறிலி uint32_t dst_row0 = i3 * ne2 * ne1;
            மாறிலி uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            மாறிலி uint32_t n2_first = ne2 - i2_src0;

            roll_dma_தள்ளு_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0);
            roll_dma_தள்ளு_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0);
        }

        dma_queue_flush(q);
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    மாறிலி uint32_t n1_first = ne1 - i1_src0;

    க்கு (uint32_t i3 = 0; i3 < ne3; i3++) {
        மாறிலி uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        க்கு (uint32_t i2 = 0; i2 < ne2; i2++) {
            மாறிலி uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            மாறிலி uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            மாறிலி uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_தள்ளு_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0);
            roll_dma_தள்ளு_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0);
        }
    }

    dma_queue_flush(q);
    திரும்பு HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
நிலையான முழு roll_dma_f32_strided(கட்டமைப்பு htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    மாறிலி uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    மாறிலி uint32_t n1_first = ne1 - i1_src0;

    க்கு (uint32_t i3 = 0; i3 < ne3; i3++) {
        மாறிலி uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        க்கு (uint32_t i2 = 0; i2 < ne2; i2++) {
            மாறிலி uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            மாறிலி dma_addr_t dst_row0 = dst->data  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            மாறிலி dma_addr_t src_row0 = src0->data + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_தள்ளு_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_தள்ளு_range(q, dst_row0 + (size_t) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    திரும்பு HTP_STATUS_OK;
}

நிலையான வெறுமை roll_thread_f32(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_roll_context * rctx = (கட்டமைப்பு htp_roll_context *) data;
    கட்டமைப்பு htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    மாறிலி uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    மாறிலி uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    எனில் (row_start >= row_end) {
        திரும்பு;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    க்கு (uint32_t row = row_start; row < row_end; row++) {
        மாறிலி uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        மாறிலி uint32_t rem = row - i3 * ne2 * ne1;
        மாறிலி uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        மாறிலி uint32_t i1  = rem - i2 * ne1;

        மாறிலி uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        மாறிலி uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        மாறிலி uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        மாறிலி uint8_t * src_row = (மாறிலி uint8_t *) (uintptr_t) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->data + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * அளவு(தசம), n0 * அளவு(தசம), ne0 * அளவு(தசம), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * அளவு(தசம), n0, அளவு(தசம));

        எனில் (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * அளவு(தசம), ne0 * அளவு(தசம), 1);
            hvx_copy_uu(dst_row + n0 * அளவு(தசம), src_row, i0_src0, அளவு(தசம));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(உயர்வு_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

முழு execute_op_roll_f32(கட்டமைப்பு htp_ops_context * octx) {
    htp_roll_preamble;

    எனில் (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (src0->nb[0] != அளவு(தசம) || dst->nb[0] != அளவு(தசம)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    மாறிலி uint32_t total_rows = ne1 * ne2 * ne3;
    மாறிலி size_t dst_row_size = ne0 * அளவு(தசம);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, அளவு(தசம), (uint32_t) dst_row_size, &rows_per_chunk);
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
    }

    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    எனில் (octx->ctx->mdev.மொத்தம் <= 1) {
        எனில் (htp_tensor_is_contiguous(src0, அளவு(தசம)) && htp_tensor_is_contiguous(dst, அளவு(தசம))) {
            திரும்பு roll_dma_f32_contiguous(octx);
        }
        திரும்பு roll_dma_f32_strided(octx);
    }

    எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி uint32_t n_threads = octx->n_threads;
    கட்டமைப்பு htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_மதிப்புகள்(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_மதிப்புகள்(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    திரும்பு HTP_STATUS_OK;
}

முழு op_roll(கட்டமைப்பு htp_ops_context * octx) {
    தேர்வு (octx->src[0]->type) {
        நிலை HTP_TYPE_F32:
            திரும்பு execute_op_roll_f32(octx);

        இயல்பு:
            திரும்பு HTP_STATUS_NO_SUPPORT;
    }
}
