#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#অন্তর্ভাব <string.h>

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hvx-utils.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"

কাঠামো htp_roll_context {
    কাঠামো htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    কাঠামো fastdiv_মানসমূহ div_ne1;
    কাঠামো fastdiv_মানসমূহ div_ne2_ne1;
};

স্থির অন্তঃরেখ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    যদি (i < 0) {
        ফেরত (uint32_t) (i + (int32_t) ne);
    }
    যদি ((uint32_t) i >= ne) {
        ফেরত (uint32_t) i - ne;
    }
    ফেরত (uint32_t) i;
}

#define htp_roll_preamble                             \
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];    \
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;       \
                                                      \
    ধ্রুবক uint32_t ne0 = dst->ne[0];                  \
    ধ্রুবক uint32_t ne1 = dst->ne[1];                  \
    ধ্রুবক uint32_t ne2 = dst->ne[2];                  \
    ধ্রুবক uint32_t ne3 = dst->ne[3];                  \
                                                      \
    ধ্রুবক uint32_t nb01 = src0->nb[1];                \
    ধ্রুবক uint32_t nb02 = src0->nb[2];                \
    ধ্রুবক uint32_t nb03 = src0->nb[3];                \
                                                      \
    ধ্রুবক uint32_t nb1 = dst->nb[1];                  \
    ধ্রুবক uint32_t nb2 = dst->nb[2];                  \
    ধ্রুবক uint32_t nb3 = dst->nb[3];                  \
                                                      \
    ধ্রুবক int32_t s0 = octx->op_params[0];            \
    ধ্রুবক int32_t s1 = octx->op_params[1];            \
    ধ্রুবক int32_t s2 = octx->op_params[2];            \
    ধ্রুবক int32_t s3 = octx->op_params[3];            \
                                                      \
    ধ্রুবক uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    ধ্রুবক uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

স্থির অন্তঃরেখ শূন্য roll_dma_ঠেলো(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    যদি (bytes == 0 || nrows == 0) {
        ফেরত;
    }

    যদি (!dma_queue_ঠেলো(q, dma_make_তথ্য(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_ঠেলো(q, dma_make_তথ্য(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

স্থির অন্তঃরেখ শূন্য roll_dma_ঠেলো_rows(dma_queue *               q,
                                      ধ্রুবক কাঠামো htp_tensor * dst,
                                      ধ্রুবক কাঠামো htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_আকার,
                                      uint32_t                  i0_src0) {
    ধ্রুবক dma_addr_t dst_base = dst->তথ্য + (size_t) dst_row * row_আকার;
    ধ্রুবক dma_addr_t src_base = src0->তথ্য + (size_t) src_row * row_আকার;
    ধ্রুবক uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ঠেলো(q, dst_base, src_base + (size_t) i0_src0 * আকার(দশমিক),
                  row_আকার, row_আকার, n0 * আকার(দশমিক), nrows);
    roll_dma_ঠেলো(q, dst_base + (size_t) n0 * আকার(দশমিক), src_base,
                  row_আকার, row_আকার, i0_src0 * আকার(দশমিক), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
স্থির অন্তঃরেখ শূন্য roll_dma_ঠেলো_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ঠেলো(q, dst_row, src_row + (size_t) i0_src0 * আকার(দশমিক),
                  dst_stride, src_stride, n0 * আকার(দশমিক), nrows);
    roll_dma_ঠেলো(q, dst_row + (size_t) n0 * আকার(দশমিক), src_row,
                  dst_stride, src_stride, i0_src0 * আকার(দশমিক), nrows);
}

স্থির পূর্ণসংখ্যা roll_dma_f32_contiguous(কাঠামো htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ধ্রুবক uint32_t row_আকার = ne0 * আকার(দশমিক);

    যদি (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ঠেলো_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_আকার, i0_src0);
        dma_queue_flush(q);
        ফেরত HTP_STATUS_OK;
    }

    যদি (s1 == 0) {
        ধ্রুবক uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        জন্য (uint32_t i3 = 0; i3 < ne3; i3++) {
            ধ্রুবক uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ধ্রুবক uint32_t dst_row0 = i3 * ne2 * ne1;
            ধ্রুবক uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ধ্রুবক uint32_t n2_first = ne2 - i2_src0;

            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_আকার, i0_src0);
            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_আকার, i0_src0);
        }

        dma_queue_flush(q);
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ধ্রুবক uint32_t n1_first = ne1 - i1_src0;

    জন্য (uint32_t i3 = 0; i3 < ne3; i3++) {
        ধ্রুবক uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        জন্য (uint32_t i2 = 0; i2 < ne2; i2++) {
            ধ্রুবক uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ধ্রুবক uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ধ্রুবক uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_আকার, i0_src0);
            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_আকার, i0_src0);
        }
    }

    dma_queue_flush(q);
    ফেরত HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
স্থির পূর্ণসংখ্যা roll_dma_f32_strided(কাঠামো htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ধ্রুবক uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ধ্রুবক uint32_t n1_first = ne1 - i1_src0;

    জন্য (uint32_t i3 = 0; i3 < ne3; i3++) {
        ধ্রুবক uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        জন্য (uint32_t i2 = 0; i2 < ne2; i2++) {
            ধ্রুবক uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ধ্রুবক dma_addr_t dst_row0 = dst->তথ্য  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ধ্রুবক dma_addr_t src_row0 = src0->তথ্য + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ঠেলো_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_ঠেলো_range(q, dst_row0 + (size_t) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    ফেরত HTP_STATUS_OK;
}

স্থির শূন্য roll_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_roll_context * rctx = (কাঠামো htp_roll_context *) তথ্য;
    কাঠামো htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ধ্রুবক uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    যদি (row_start >= row_end) {
        ফেরত;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    জন্য (uint32_t row = row_start; row < row_end; row++) {
        ধ্রুবক uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        ধ্রুবক uint32_t rem = row - i3 * ne2 * ne1;
        ধ্রুবক uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ধ্রুবক uint32_t i1  = rem - i2 * ne1;

        ধ্রুবক uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ধ্রুবক uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ধ্রুবক uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ধ্রুবক uint8_t * src_row = (ধ্রুবক uint8_t *) (uintptr_t) src0->তথ্য + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->তথ্য + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * আকার(দশমিক), n0 * আকার(দশমিক), ne0 * আকার(দশমিক), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * আকার(দশমিক), n0, আকার(দশমিক));

        যদি (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * আকার(দশমিক), ne0 * আকার(দশমিক), 1);
            hvx_copy_uu(dst_row + n0 * আকার(দশমিক), src_row, i0_src0, আকার(দশমিক));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(উপরের_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

পূর্ণসংখ্যা execute_op_roll_f32(কাঠামো htp_ops_context * octx) {
    htp_roll_preamble;

    যদি (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (src0->nb[0] != আকার(দশমিক) || dst->nb[0] != আকার(দশমিক)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুবক uint32_t যোগ_rows = ne1 * ne2 * ne3;
    ধ্রুবক size_t dst_row_আকার = ne0 * আকার(দশমিক);

    uint32_t row_start = 0;
    uint32_t nrows     = যোগ_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, আকার(দশমিক), (uint32_t) dst_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    যদি (octx->ctx->mdev.গুনতি <= 1) {
        যদি (htp_tensor_is_contiguous(src0, আকার(দশমিক)) && htp_tensor_is_contiguous(dst, আকার(দশমিক))) {
            ফেরত roll_dma_f32_contiguous(octx);
        }
        ফেরত roll_dma_f32_strided(octx);
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;
    কাঠামো htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_মানসমূহ(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_মানসমূহ(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_roll(কাঠামো htp_ops_context * octx) {
    নির্বাচন (octx->src[0]->type) {
        ক্ষেত্র HTP_TYPE_F32:
            ফেরত execute_op_roll_f32(octx);

        পূর্বনির্ধারিত:
            ফেরত HTP_STATUS_NO_SUPPORT;
    }
}
