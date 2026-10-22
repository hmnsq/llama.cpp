#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"

ସଂରଚନା htp_roll_context {
    ସଂରଚନା htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne1;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne2_ne1;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    ଯଦି (i < 0) {
        ଫେରନ୍ତୁ (uint32_t) (i + (int32_t) ne);
    }
    ଯଦି ((uint32_t) i >= ne) {
        ଫେରନ୍ତୁ (uint32_t) i - ne;
    }
    ଫେରନ୍ତୁ (uint32_t) i;
}

#define htp_roll_preamble                             \
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];    \
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;       \
                                                      \
    ଅଚଳ uint32_t ne0 = dst->ne[0];                  \
    ଅଚଳ uint32_t ne1 = dst->ne[1];                  \
    ଅଚଳ uint32_t ne2 = dst->ne[2];                  \
    ଅଚଳ uint32_t ne3 = dst->ne[3];                  \
                                                      \
    ଅଚଳ uint32_t nb01 = src0->nb[1];                \
    ଅଚଳ uint32_t nb02 = src0->nb[2];                \
    ଅଚଳ uint32_t nb03 = src0->nb[3];                \
                                                      \
    ଅଚଳ uint32_t nb1 = dst->nb[1];                  \
    ଅଚଳ uint32_t nb2 = dst->nb[2];                  \
    ଅଚଳ uint32_t nb3 = dst->nb[3];                  \
                                                      \
    ଅଚଳ int32_t s0 = octx->op_params[0];            \
    ଅଚଳ int32_t s1 = octx->op_params[1];            \
    ଅଚଳ int32_t s2 = octx->op_params[2];            \
    ଅଚଳ int32_t s3 = octx->op_params[3];            \
                                                      \
    ଅଚଳ uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    ଅଚଳ uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ roll_dma_ଠେଲ(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    ଯଦି (bytes == 0 || nrows == 0) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (!dma_queue_ଠେଲ(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_ଠେଲ(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ roll_dma_ଠେଲ_rows(dma_queue *               q,
                                      ଅଚଳ ସଂରଚନା htp_tensor * dst,
                                      ଅଚଳ ସଂରଚନା htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0) {
    ଅଚଳ dma_addr_t dst_base = dst->data + (size_t) dst_row * row_size;
    ଅଚଳ dma_addr_t src_base = src0->data + (size_t) src_row * row_size;
    ଅଚଳ uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ଠେଲ(q, dst_base, src_base + (size_t) i0_src0 * ଆକାର(ଦଶମିକ),
                  row_size, row_size, n0 * ଆକାର(ଦଶମିକ), nrows);
    roll_dma_ଠେଲ(q, dst_base + (size_t) n0 * ଆକାର(ଦଶମିକ), src_base,
                  row_size, row_size, i0_src0 * ଆକାର(ଦଶମିକ), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ roll_dma_ଠେଲ_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ଠେଲ(q, dst_row, src_row + (size_t) i0_src0 * ଆକାର(ଦଶମିକ),
                  dst_stride, src_stride, n0 * ଆକାର(ଦଶମିକ), nrows);
    roll_dma_ଠେଲ(q, dst_row + (size_t) n0 * ଆକାର(ଦଶମିକ), src_row,
                  dst_stride, src_stride, i0_src0 * ଆକାର(ଦଶମିକ), nrows);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା roll_dma_f32_contiguous(ସଂରଚନା htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ଅଚଳ uint32_t row_size = ne0 * ଆକାର(ଦଶମିକ);

    ଯଦି (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ଠେଲ_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0);
        dma_queue_flush(q);
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଯଦି (s1 == 0) {
        ଅଚଳ uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        ପାଇଁ (uint32_t i3 = 0; i3 < ne3; i3++) {
            ଅଚଳ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ଅଚଳ uint32_t dst_row0 = i3 * ne2 * ne1;
            ଅଚଳ uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ଅଚଳ uint32_t n2_first = ne2 - i2_src0;

            roll_dma_ଠେଲ_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0);
            roll_dma_ଠେଲ_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0);
        }

        dma_queue_flush(q);
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ଅଚଳ uint32_t n1_first = ne1 - i1_src0;

    ପାଇଁ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ଅଚଳ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ପାଇଁ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ଅଚଳ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ଅଚଳ uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ଅଚଳ uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ଠେଲ_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0);
            roll_dma_ଠେଲ_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0);
        }
    }

    dma_queue_flush(q);
    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା roll_dma_f32_strided(ସଂରଚନା htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ଅଚଳ uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ଅଚଳ uint32_t n1_first = ne1 - i1_src0;

    ପାଇଁ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ଅଚଳ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ପାଇଁ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ଅଚଳ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ଅଚଳ dma_addr_t dst_row0 = dst->data  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ଅଚଳ dma_addr_t src_row0 = src0->data + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ଠେଲ_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_ଠେଲ_range(q, dst_row0 + (size_t) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ଶୂନ୍ୟ roll_thread_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_roll_context * rctx = (ସଂରଚନା htp_roll_context *) data;
    ସଂରଚନା htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ଅଚଳ uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ଅଚଳ uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    ଯଦି (row_start >= row_end) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    ପାଇଁ (uint32_t row = row_start; row < row_end; row++) {
        ଅଚଳ uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        ଅଚଳ uint32_t rem = row - i3 * ne2 * ne1;
        ଅଚଳ uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ଅଚଳ uint32_t i1  = rem - i2 * ne1;

        ଅଚଳ uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ଅଚଳ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ଅଚଳ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ଅଚଳ uint8_t * src_row = (ଅଚଳ uint8_t *) (uintptr_t) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->data + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * ଆକାର(ଦଶମିକ), n0 * ଆକାର(ଦଶମିକ), ne0 * ଆକାର(ଦଶମିକ), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * ଆକାର(ଦଶମିକ), n0, ଆକାର(ଦଶମିକ));

        ଯଦି (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * ଆକାର(ଦଶମିକ), ne0 * ଆକାର(ଦଶମିକ), 1);
            hvx_copy_uu(dst_row + n0 * ଆକାର(ଦଶମିକ), src_row, i0_src0, ଆକାର(ଦଶମିକ));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(ଉପର_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା execute_op_roll_f32(ସଂରଚନା htp_ops_context * octx) {
    htp_roll_preamble;

    ଯଦି (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (src0->nb[0] != ଆକାର(ଦଶମିକ) || dst->nb[0] != ଆକାର(ଦଶମିକ)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଅଚଳ uint32_t total_rows = ne1 * ne2 * ne3;
    ଅଚଳ size_t dst_row_size = ne0 * ଆକାର(ଦଶମିକ);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ଆକାର(ଦଶମିକ), (uint32_t) dst_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 <= 1) {
        ଯଦି (htp_tensor_is_contiguous(src0, ଆକାର(ଦଶମିକ)) && htp_tensor_is_contiguous(dst, ଆକାର(ଦଶମିକ))) {
            ଫେରନ୍ତୁ roll_dma_f32_contiguous(octx);
        }
        ଫେରନ୍ତୁ roll_dma_f32_strided(octx);
    }

    ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;
    ସଂରଚନା htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_roll(ସଂରଚନା htp_ops_context * octx) {
    ବାଛ (octx->src[0]->type) {
        ସ୍ଥିତି HTP_TYPE_F32:
            ଫେରନ୍ତୁ execute_op_roll_f32(octx);

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }
}
