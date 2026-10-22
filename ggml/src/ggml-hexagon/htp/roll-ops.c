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

    संरचना fastdiv_मूल्ये div_ne1;
    संरचना fastdiv_मूल्ये div_ne2_ne1;
};

स्थिर ओळीत uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    जर (i < 0) {
        परत (uint32_t) (i + (int32_t) ne);
    }
    जर ((uint32_t) i >= ne) {
        परत (uint32_t) i - ne;
    }
    परत (uint32_t) i;
}

#define htp_roll_preamble                             \
    स्थिरांक संरचना htp_tensor * src0 = octx->src[0];    \
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;       \
                                                      \
    स्थिरांक uint32_t ne0 = dst->ne[0];                  \
    स्थिरांक uint32_t ne1 = dst->ne[1];                  \
    स्थिरांक uint32_t ne2 = dst->ne[2];                  \
    स्थिरांक uint32_t ne3 = dst->ne[3];                  \
                                                      \
    स्थिरांक uint32_t nb01 = src0->nb[1];                \
    स्थिरांक uint32_t nb02 = src0->nb[2];                \
    स्थिरांक uint32_t nb03 = src0->nb[3];                \
                                                      \
    स्थिरांक uint32_t nb1 = dst->nb[1];                  \
    स्थिरांक uint32_t nb2 = dst->nb[2];                  \
    स्थिरांक uint32_t nb3 = dst->nb[3];                  \
                                                      \
    स्थिरांक int32_t s0 = octx->op_params[0];            \
    स्थिरांक int32_t s1 = octx->op_params[1];            \
    स्थिरांक int32_t s2 = octx->op_params[2];            \
    स्थिरांक int32_t s3 = octx->op_params[3];            \
                                                      \
    स्थिरांक uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    स्थिरांक uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

स्थिर ओळीत शून्य roll_dma_ढकला(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    जर (bytes == 0 || nrows == 0) {
        परत;
    }

    जर (!dma_queue_ढकला(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_ढकला(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

स्थिर ओळीत शून्य roll_dma_ढकला_rows(dma_queue *               q,
                                      स्थिरांक संरचना htp_tensor * dst,
                                      स्थिरांक संरचना htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0) {
    स्थिरांक dma_addr_t dst_base = dst->data + (माप_प्रकार) dst_row * row_size;
    स्थिरांक dma_addr_t src_base = src0->data + (माप_प्रकार) src_row * row_size;
    स्थिरांक uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ढकला(q, dst_base, src_base + (माप_प्रकार) i0_src0 * आकार(दशांश),
                  row_size, row_size, n0 * आकार(दशांश), nrows);
    roll_dma_ढकला(q, dst_base + (माप_प्रकार) n0 * आकार(दशांश), src_base,
                  row_size, row_size, i0_src0 * आकार(दशांश), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
स्थिर ओळीत शून्य roll_dma_ढकला_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ढकला(q, dst_row, src_row + (माप_प्रकार) i0_src0 * आकार(दशांश),
                  dst_stride, src_stride, n0 * आकार(दशांश), nrows);
    roll_dma_ढकला(q, dst_row + (माप_प्रकार) n0 * आकार(दशांश), src_row,
                  dst_stride, src_stride, i0_src0 * आकार(दशांश), nrows);
}

स्थिर पूर्णांक roll_dma_f32_contiguous(संरचना htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    स्थिरांक uint32_t row_size = ne0 * आकार(दशांश);

    जर (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ढकला_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0);
        dma_queue_flush(q);
        परत HTP_STATUS_OK;
    }

    जर (s1 == 0) {
        स्थिरांक uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        साठी (uint32_t i3 = 0; i3 < ne3; i3++) {
            स्थिरांक uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            स्थिरांक uint32_t dst_row0 = i3 * ne2 * ne1;
            स्थिरांक uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            स्थिरांक uint32_t n2_first = ne2 - i2_src0;

            roll_dma_ढकला_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0);
            roll_dma_ढकला_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0);
        }

        dma_queue_flush(q);
        परत HTP_STATUS_OK;
    }

    स्थिरांक uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    स्थिरांक uint32_t n1_first = ne1 - i1_src0;

    साठी (uint32_t i3 = 0; i3 < ne3; i3++) {
        स्थिरांक uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        साठी (uint32_t i2 = 0; i2 < ne2; i2++) {
            स्थिरांक uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            स्थिरांक uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            स्थिरांक uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ढकला_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0);
            roll_dma_ढकला_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0);
        }
    }

    dma_queue_flush(q);
    परत HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
स्थिर पूर्णांक roll_dma_f32_strided(संरचना htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    स्थिरांक uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    स्थिरांक uint32_t n1_first = ne1 - i1_src0;

    साठी (uint32_t i3 = 0; i3 < ne3; i3++) {
        स्थिरांक uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        साठी (uint32_t i2 = 0; i2 < ne2; i2++) {
            स्थिरांक uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            स्थिरांक dma_addr_t dst_row0 = dst->data  + (माप_प्रकार) i2  * nb2  + (माप_प्रकार) i3  * nb3;
            स्थिरांक dma_addr_t src_row0 = src0->data + (माप_प्रकार) i02 * nb02 + (माप_प्रकार) i03 * nb03;

            roll_dma_ढकला_range(q, dst_row0, src_row0 + (माप_प्रकार) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_ढकला_range(q, dst_row0 + (माप_प्रकार) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    परत HTP_STATUS_OK;
}

स्थिर शून्य roll_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {
    संरचना htp_roll_context * rctx = (संरचना htp_roll_context *) data;
    संरचना htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    स्थिरांक uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    स्थिरांक uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    जर (row_start >= row_end) {
        परत;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    साठी (uint32_t row = row_start; row < row_end; row++) {
        स्थिरांक uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        स्थिरांक uint32_t rem = row - i3 * ne2 * ne1;
        स्थिरांक uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        स्थिरांक uint32_t i1  = rem - i2 * ne1;

        स्थिरांक uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        स्थिरांक uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        स्थिरांक uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        स्थिरांक uint8_t * src_row = (स्थिरांक uint8_t *) (uintptr_t) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->data + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * आकार(दशांश), n0 * आकार(दशांश), ne0 * आकार(दशांश), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * आकार(दशांश), n0, आकार(दशांश));

        जर (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * आकार(दशांश), ne0 * आकार(दशांश), 1);
            hvx_copy_uu(dst_row + n0 * आकार(दशांश), src_row, i0_src0, आकार(दशांश));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(वरचा_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

पूर्णांक execute_op_roll_f32(संरचना htp_ops_context * octx) {
    htp_roll_preamble;

    जर (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (src0->nb[0] != आकार(दशांश) || dst->nb[0] != आकार(दशांश)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        परत HTP_STATUS_INVAL_PARAMS;
    }

    स्थिरांक uint32_t total_rows = ne1 * ne2 * ne3;
    स्थिरांक माप_प्रकार dst_row_size = ne0 * आकार(दशांश);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    जर (octx->ctx->mdev.मोजणी > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, आकार(दशांश), (uint32_t) dst_row_size, &rows_per_chunk);
        स्थिरांक संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.मोजणी, &octx->ctx->mdev.मोजणी_div);
        row_start = range.start;
        nrows     = range.मोजणी;
    }

    जर (nrows == 0) {
        परत HTP_STATUS_OK;
    }

    जर (octx->ctx->mdev.मोजणी <= 1) {
        जर (htp_tensor_is_contiguous(src0, आकार(दशांश)) && htp_tensor_is_contiguous(dst, आकार(दशांश))) {
            परत roll_dma_f32_contiguous(octx);
        }
        परत roll_dma_f32_strided(octx);
    }

    जर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    स्थिरांक uint32_t n_threads = octx->n_threads;
    संरचना htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_मूल्ये(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_मूल्ये(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    परत HTP_STATUS_OK;
}

पूर्णांक op_roll(संरचना htp_ops_context * octx) {
    निवड (octx->src[0]->type) {
        स्थिती HTP_TYPE_F32:
            परत execute_op_roll_f32(octx);

        पूर्वनिर्धारित:
            परत HTP_STATUS_NO_SUPPORT;
    }
}
