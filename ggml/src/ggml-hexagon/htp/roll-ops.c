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

সংরচনা htp_roll_context {
    সংরচনা htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    সংরচনা fastdiv_মানসমূহ div_ne1;
    সংরচনা fastdiv_মানসমূহ div_ne2_ne1;
};

স্থির অন্তঃস্থ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    যদি (i < 0) {
        প্রত্যাগম (uint32_t) (i + (int32_t) ne);
    }
    যদি ((uint32_t) i >= ne) {
        প্রত্যাগম (uint32_t) i - ne;
    }
    প্রত্যাগম (uint32_t) i;
}

#define htp_roll_preamble                             \
    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];    \
    ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;       \
                                                      \
    ধ্রুব uint32_t ne0 = dst->ne[0];                  \
    ধ্রুব uint32_t ne1 = dst->ne[1];                  \
    ধ্রুব uint32_t ne2 = dst->ne[2];                  \
    ধ্রুব uint32_t ne3 = dst->ne[3];                  \
                                                      \
    ধ্রুব uint32_t nb01 = src0->nb[1];                \
    ধ্রুব uint32_t nb02 = src0->nb[2];                \
    ধ্রুব uint32_t nb03 = src0->nb[3];                \
                                                      \
    ধ্রুব uint32_t nb1 = dst->nb[1];                  \
    ধ্রুব uint32_t nb2 = dst->nb[2];                  \
    ধ্রুব uint32_t nb3 = dst->nb[3];                  \
                                                      \
    ধ্রুব int32_t s0 = octx->op_params[0];            \
    ধ্রুব int32_t s1 = octx->op_params[1];            \
    ধ্রুব int32_t s2 = octx->op_params[2];            \
    ধ্রুব int32_t s3 = octx->op_params[3];            \
                                                      \
    ধ্রুব uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    ধ্রুব uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

স্থির অন্তঃস্থ শূন্য roll_dma_ঠেলো(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    যদি (bytes == 0 || nrows == 0) {
        প্রত্যাগম;
    }

    যদি (!dma_queue_ঠেলো(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_ঠেলো(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

স্থির অন্তঃস্থ শূন্য roll_dma_ঠেলো_rows(dma_queue *               q,
                                      ধ্রুব সংরচনা htp_tensor * dst,
                                      ধ্রুব সংরচনা htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0) {
    ধ্রুব dma_addr_t dst_base = dst->data + (size_t) dst_row * row_size;
    ধ্রুব dma_addr_t src_base = src0->data + (size_t) src_row * row_size;
    ধ্রুব uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ঠেলো(q, dst_base, src_base + (size_t) i0_src0 * পরিমাণ(ভিন্ন),
                  row_size, row_size, n0 * পরিমাণ(ভিন্ন), nrows);
    roll_dma_ঠেলো(q, dst_base + (size_t) n0 * পরিমাণ(ভিন্ন), src_base,
                  row_size, row_size, i0_src0 * পরিমাণ(ভিন্ন), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
স্থির অন্তঃস্থ শূন্য roll_dma_ঠেলো_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ঠেলো(q, dst_row, src_row + (size_t) i0_src0 * পরিমাণ(ভিন্ন),
                  dst_stride, src_stride, n0 * পরিমাণ(ভিন্ন), nrows);
    roll_dma_ঠেলো(q, dst_row + (size_t) n0 * পরিমাণ(ভিন্ন), src_row,
                  dst_stride, src_stride, i0_src0 * পরিমাণ(ভিন্ন), nrows);
}

স্থির পূর্ণাঙ্ক roll_dma_f32_contiguous(সংরচনা htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ধ্রুব uint32_t row_size = ne0 * পরিমাণ(ভিন্ন);

    যদি (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ঠেলো_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0);
        dma_queue_flush(q);
        প্রত্যাগম HTP_STATUS_OK;
    }

    যদি (s1 == 0) {
        ধ্রুব uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        প্রতি (uint32_t i3 = 0; i3 < ne3; i3++) {
            ধ্রুব uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ধ্রুব uint32_t dst_row0 = i3 * ne2 * ne1;
            ধ্রুব uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ধ্রুব uint32_t n2_first = ne2 - i2_src0;

            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0);
            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0);
        }

        dma_queue_flush(q);
        প্রত্যাগম HTP_STATUS_OK;
    }

    ধ্রুব uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ধ্রুব uint32_t n1_first = ne1 - i1_src0;

    প্রতি (uint32_t i3 = 0; i3 < ne3; i3++) {
        ধ্রুব uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        প্রতি (uint32_t i2 = 0; i2 < ne2; i2++) {
            ধ্রুব uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ধ্রুব uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ধ্রুব uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0);
            roll_dma_ঠেলো_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0);
        }
    }

    dma_queue_flush(q);
    প্রত্যাগম HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
স্থির পূর্ণাঙ্ক roll_dma_f32_strided(সংরচনা htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ধ্রুব uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ধ্রুব uint32_t n1_first = ne1 - i1_src0;

    প্রতি (uint32_t i3 = 0; i3 < ne3; i3++) {
        ধ্রুব uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        প্রতি (uint32_t i2 = 0; i2 < ne2; i2++) {
            ধ্রুব uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ধ্রুব dma_addr_t dst_row0 = dst->data  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ধ্রুব dma_addr_t src_row0 = src0->data + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ঠেলো_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_ঠেলো_range(q, dst_row0 + (size_t) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    প্রত্যাগম HTP_STATUS_OK;
}

স্থির শূন্য roll_thread_f32(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_roll_context * rctx = (সংরচনা htp_roll_context *) data;
    সংরচনা htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ধ্রুব uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ধ্রুব uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    যদি (row_start >= row_end) {
        প্রত্যাগম;
    }

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    প্রতি (uint32_t row = row_start; row < row_end; row++) {
        ধ্রুব uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        ধ্রুব uint32_t rem = row - i3 * ne2 * ne1;
        ধ্রুব uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ধ্রুব uint32_t i1  = rem - i2 * ne1;

        ধ্রুব uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ধ্রুব uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ধ্রুব uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ধ্রুব uint8_t * src_row = (ধ্রুব uint8_t *) (uintptr_t) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->data + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * পরিমাণ(ভিন্ন), n0 * পরিমাণ(ভিন্ন), ne0 * পরিমাণ(ভিন্ন), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * পরিমাণ(ভিন্ন), n0, পরিমাণ(ভিন্ন));

        যদি (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * পরিমাণ(ভিন্ন), ne0 * পরিমাণ(ভিন্ন), 1);
            hvx_copy_uu(dst_row + n0 * পরিমাণ(ভিন্ন), src_row, i0_src0, পরিমাণ(ভিন্ন));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(উপরের_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

পূর্ণাঙ্ক execute_op_roll_f32(সংরচনা htp_ops_context * octx) {
    htp_roll_preamble;

    যদি (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    যদি (src0->nb[0] != পরিমাণ(ভিন্ন) || dst->nb[0] != পরিমাণ(ভিন্ন)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    যদি (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুব uint32_t total_rows = ne1 * ne2 * ne3;
    ধ্রুব size_t dst_row_size = ne0 * পরিমাণ(ভিন্ন);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, পরিমাণ(ভিন্ন), (uint32_t) dst_row_size, &rows_per_chunk);
        ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }

    যদি (octx->ctx->mdev.গুনতি <= 1) {
        যদি (htp_tensor_is_contiguous(src0, পরিমাণ(ভিন্ন)) && htp_tensor_is_contiguous(dst, পরিমাণ(ভিন্ন))) {
            প্রত্যাগম roll_dma_f32_contiguous(octx);
        }
        প্রত্যাগম roll_dma_f32_strided(octx);
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব uint32_t n_threads = octx->n_threads;
    সংরচনা htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_মানসমূহ(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_মানসমূহ(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    প্রত্যাগম HTP_STATUS_OK;
}

পূর্ণাঙ্ক op_roll(সংরচনা htp_ops_context * octx) {
    বিকল্প (octx->src[0]->type) {
        পক্ষ HTP_TYPE_F32:
            প্রত্যাগম execute_op_roll_f32(octx);

        স্বভাব:
            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }
}
