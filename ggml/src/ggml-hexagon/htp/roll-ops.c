#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

ಸಂರಚನಾ htp_roll_context {
    ಸಂರಚನಾ htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne1;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne2_ne1;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    ಯದಿ (i < 0) {
        ಪ್ರತ್ಯಾಗಮ (uint32_t) (i + (int32_t) ne);
    }
    ಯದಿ ((uint32_t) i >= ne) {
        ಪ್ರತ್ಯಾಗಮ (uint32_t) i - ne;
    }
    ಪ್ರತ್ಯಾಗಮ (uint32_t) i;
}

#define htp_roll_preamble                             \
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];    \
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;       \
                                                      \
    ಧ್ರುವ uint32_t ne0 = dst->ne[0];                  \
    ಧ್ರುವ uint32_t ne1 = dst->ne[1];                  \
    ಧ್ರುವ uint32_t ne2 = dst->ne[2];                  \
    ಧ್ರುವ uint32_t ne3 = dst->ne[3];                  \
                                                      \
    ಧ್ರುವ uint32_t nb01 = src0->nb[1];                \
    ಧ್ರುವ uint32_t nb02 = src0->nb[2];                \
    ಧ್ರುವ uint32_t nb03 = src0->nb[3];                \
                                                      \
    ಧ್ರುವ uint32_t nb1 = dst->nb[1];                  \
    ಧ್ರುವ uint32_t nb2 = dst->nb[2];                  \
    ಧ್ರುವ uint32_t nb3 = dst->nb[3];                  \
                                                      \
    ಧ್ರುವ int32_t s0 = octx->op_params[0];            \
    ಧ್ರುವ int32_t s1 = octx->op_params[1];            \
    ಧ್ರುವ int32_t s2 = octx->op_params[2];            \
    ಧ್ರುವ int32_t s3 = octx->op_params[3];            \
                                                      \
    ಧ್ರುವ uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    ಧ್ರುವ uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ roll_dma_ತಳ್ಳು(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    ಯದಿ (bytes == 0 || nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (!dma_queue_ತಳ್ಳು(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_ತಳ್ಳು(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ roll_dma_ತಳ್ಳು_rows(dma_queue *               q,
                                      ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst,
                                      ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0) {
    ಧ್ರುವ dma_addr_t dst_base = dst->data + (size_t) dst_row * row_size;
    ಧ್ರುವ dma_addr_t src_base = src0->data + (size_t) src_row * row_size;
    ಧ್ರುವ uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ತಳ್ಳು(q, dst_base, src_base + (size_t) i0_src0 * ಪರಿಮಾಣ(ಭಿನ್ನ),
                  row_size, row_size, n0 * ಪರಿಮಾಣ(ಭಿನ್ನ), nrows);
    roll_dma_ತಳ್ಳು(q, dst_base + (size_t) n0 * ಪರಿಮಾಣ(ಭಿನ್ನ), src_base,
                  row_size, row_size, i0_src0 * ಪರಿಮಾಣ(ಭಿನ್ನ), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ roll_dma_ತಳ್ಳು_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ತಳ್ಳು(q, dst_row, src_row + (size_t) i0_src0 * ಪರಿಮಾಣ(ಭಿನ್ನ),
                  dst_stride, src_stride, n0 * ಪರಿಮಾಣ(ಭಿನ್ನ), nrows);
    roll_dma_ತಳ್ಳು(q, dst_row + (size_t) n0 * ಪರಿಮಾಣ(ಭಿನ್ನ), src_row,
                  dst_stride, src_stride, i0_src0 * ಪರಿಮಾಣ(ಭಿನ್ನ), nrows);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ roll_dma_f32_contiguous(ಸಂರಚನಾ htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ಧ್ರುವ uint32_t row_size = ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಯದಿ (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ತಳ್ಳು_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0);
        dma_queue_flush(q);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಯದಿ (s1 == 0) {
        ಧ್ರುವ uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        ಪ್ರತಿ (uint32_t i3 = 0; i3 < ne3; i3++) {
            ಧ್ರುವ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ಧ್ರುವ uint32_t dst_row0 = i3 * ne2 * ne1;
            ಧ್ರುವ uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ಧ್ರುವ uint32_t n2_first = ne2 - i2_src0;

            roll_dma_ತಳ್ಳು_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0);
            roll_dma_ತಳ್ಳು_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0);
        }

        dma_queue_flush(q);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ಧ್ರುವ uint32_t n1_first = ne1 - i1_src0;

    ಪ್ರತಿ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ಧ್ರುವ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ಪ್ರತಿ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ಧ್ರುವ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ಧ್ರುವ uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ಧ್ರುವ uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ತಳ್ಳು_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0);
            roll_dma_ತಳ್ಳು_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0);
        }
    }

    dma_queue_flush(q);
    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ roll_dma_f32_strided(ಸಂರಚನಾ htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ಧ್ರುವ uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ಧ್ರುವ uint32_t n1_first = ne1 - i1_src0;

    ಪ್ರತಿ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ಧ್ರುವ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ಪ್ರತಿ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ಧ್ರುವ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ಧ್ರುವ dma_addr_t dst_row0 = dst->data  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ಧ್ರುವ dma_addr_t src_row0 = src0->data + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ತಳ್ಳು_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_ತಳ್ಳು_range(q, dst_row0 + (size_t) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಸ್ಥಿರ ಶೂನ್ಯ roll_thread_f32(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_roll_context * rctx = (ಸಂರಚನಾ htp_roll_context *) data;
    ಸಂರಚನಾ htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ಧ್ರುವ uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ಧ್ರುವ uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    ಯದಿ (row_start >= row_end) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    ಪ್ರತಿ (uint32_t row = row_start; row < row_end; row++) {
        ಧ್ರುವ uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        ಧ್ರುವ uint32_t rem = row - i3 * ne2 * ne1;
        ಧ್ರುವ uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ಧ್ರುವ uint32_t i1  = rem - i2 * ne1;

        ಧ್ರುವ uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ಧ್ರುವ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ಧ್ರುವ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ಧ್ರುವ uint8_t * src_row = (ಧ್ರುವ uint8_t *) (uintptr_t) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->data + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * ಪರಿಮಾಣ(ಭಿನ್ನ), n0 * ಪರಿಮಾಣ(ಭಿನ್ನ), ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * ಪರಿಮಾಣ(ಭಿನ್ನ), n0, ಪರಿಮಾಣ(ಭಿನ್ನ));

        ಯದಿ (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * ಪರಿಮಾಣ(ಭಿನ್ನ), ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ), 1);
            hvx_copy_uu(dst_row + n0 * ಪರಿಮಾಣ(ಭಿನ್ನ), src_row, i0_src0, ಪರಿಮಾಣ(ಭಿನ್ನ));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(ಮೇಲಿನ_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

ಪೂರ್ಣಾಙ್ಕ execute_op_roll_f32(ಸಂರಚನಾ htp_ops_context * octx) {
    htp_roll_preamble;

    ಯದಿ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (src0->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || dst->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }

    ಧ್ರುವ uint32_t total_rows = ne1 * ne2 * ne3;
    ಧ್ರುವ size_t dst_row_size = ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ಪರಿಮಾಣ(ಭಿನ್ನ), (uint32_t) dst_row_size, &rows_per_chunk);
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_start = range.start;
        nrows     = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ <= 1) {
        ಯದಿ (htp_tensor_is_contiguous(src0, ಪರಿಮಾಣ(ಭಿನ್ನ)) && htp_tensor_is_contiguous(dst, ಪರಿಮಾಣ(ಭಿನ್ನ))) {
            ಪ್ರತ್ಯಾಗಮ roll_dma_f32_contiguous(octx);
        }
        ಪ್ರತ್ಯಾಗಮ roll_dma_f32_strided(octx);
    }

    ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಧ್ರುವ uint32_t n_threads = octx->n_threads;
    ಸಂರಚನಾ htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಪೂರ್ಣಾಙ್ಕ op_roll(ಸಂರಚನಾ htp_ops_context * octx) {
    ವಿಕಲ್ಪ (octx->src[0]->type) {
        ಪಕ್ಷ HTP_TYPE_F32:
            ಪ್ರತ್ಯಾಗಮ execute_op_roll_f32(octx);

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }
}
