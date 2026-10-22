#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>

#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"

ഘടന htp_roll_context {
    ഘടന htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne1;
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne2_ne1;
};

സ്ഥിതം വരിയിൽ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    എങ്കിൽ (i < 0) {
        തിരികെ (uint32_t) (i + (int32_t) ne);
    }
    എങ്കിൽ ((uint32_t) i >= ne) {
        തിരികെ (uint32_t) i - ne;
    }
    തിരികെ (uint32_t) i;
}

#define htp_roll_preamble                             \
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];    \
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;       \
                                                      \
    സ്ഥിരം uint32_t ne0 = dst->ne[0];                  \
    സ്ഥിരം uint32_t ne1 = dst->ne[1];                  \
    സ്ഥിരം uint32_t ne2 = dst->ne[2];                  \
    സ്ഥിരം uint32_t ne3 = dst->ne[3];                  \
                                                      \
    സ്ഥിരം uint32_t nb01 = src0->nb[1];                \
    സ്ഥിരം uint32_t nb02 = src0->nb[2];                \
    സ്ഥിരം uint32_t nb03 = src0->nb[3];                \
                                                      \
    സ്ഥിരം uint32_t nb1 = dst->nb[1];                  \
    സ്ഥിരം uint32_t nb2 = dst->nb[2];                  \
    സ്ഥിരം uint32_t nb3 = dst->nb[3];                  \
                                                      \
    സ്ഥിരം int32_t s0 = octx->op_params[0];            \
    സ്ഥിരം int32_t s1 = octx->op_params[1];            \
    സ്ഥിരം int32_t s2 = octx->op_params[2];            \
    സ്ഥിരം int32_t s3 = octx->op_params[3];            \
                                                      \
    സ്ഥിരം uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    സ്ഥിരം uint32_t n0      = ne0 - i0_src0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

സ്ഥിതം വരിയിൽ ശൂന്യം roll_dma_തള്ളുക(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    എങ്കിൽ (bytes == 0 || nrows == 0) {
        തിരികെ;
    }

    എങ്കിൽ (!dma_queue_തള്ളുക(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_തള്ളുക(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം roll_dma_തള്ളുക_rows(dma_queue *               q,
                                      സ്ഥിരം ഘടന htp_tensor * dst,
                                      സ്ഥിരം ഘടന htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0) {
    സ്ഥിരം dma_addr_t dst_base = dst->data + (size_t) dst_row * row_size;
    സ്ഥിരം dma_addr_t src_base = src0->data + (size_t) src_row * row_size;
    സ്ഥിരം uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_തള്ളുക(q, dst_base, src_base + (size_t) i0_src0 * വലുപ്പം(ദശാംശം),
                  row_size, row_size, n0 * വലുപ്പം(ദശാംശം), nrows);
    roll_dma_തള്ളുക(q, dst_base + (size_t) n0 * വലുപ്പം(ദശാംശം), src_base,
                  row_size, row_size, i0_src0 * വലുപ്പം(ദശാംശം), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
സ്ഥിതം വരിയിൽ ശൂന്യം roll_dma_തള്ളുക_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_തള്ളുക(q, dst_row, src_row + (size_t) i0_src0 * വലുപ്പം(ദശാംശം),
                  dst_stride, src_stride, n0 * വലുപ്പം(ദശാംശം), nrows);
    roll_dma_തള്ളുക(q, dst_row + (size_t) n0 * വലുപ്പം(ദശാംശം), src_row,
                  dst_stride, src_stride, i0_src0 * വലുപ്പം(ദശാംശം), nrows);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ roll_dma_f32_contiguous(ഘടന htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    സ്ഥിരം uint32_t row_size = ne0 * വലുപ്പം(ദശാംശം);

    എങ്കിൽ (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_തള്ളുക_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0);
        dma_queue_flush(q);
        തിരികെ HTP_STATUS_OK;
    }

    എങ്കിൽ (s1 == 0) {
        സ്ഥിരം uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        വേണ്ടി (uint32_t i3 = 0; i3 < ne3; i3++) {
            സ്ഥിരം uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            സ്ഥിരം uint32_t dst_row0 = i3 * ne2 * ne1;
            സ്ഥിരം uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            സ്ഥിരം uint32_t n2_first = ne2 - i2_src0;

            roll_dma_തള്ളുക_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0);
            roll_dma_തള്ളുക_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0);
        }

        dma_queue_flush(q);
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    സ്ഥിരം uint32_t n1_first = ne1 - i1_src0;

    വേണ്ടി (uint32_t i3 = 0; i3 < ne3; i3++) {
        സ്ഥിരം uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        വേണ്ടി (uint32_t i2 = 0; i2 < ne2; i2++) {
            സ്ഥിരം uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            സ്ഥിരം uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            സ്ഥിരം uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_തള്ളുക_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0);
            roll_dma_തള്ളുക_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0);
        }
    }

    dma_queue_flush(q);
    തിരികെ HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
സ്ഥിതം പൂർണ്ണസംഖ്യ roll_dma_f32_strided(ഘടന htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    സ്ഥിരം uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    സ്ഥിരം uint32_t n1_first = ne1 - i1_src0;

    വേണ്ടി (uint32_t i3 = 0; i3 < ne3; i3++) {
        സ്ഥിരം uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        വേണ്ടി (uint32_t i2 = 0; i2 < ne2; i2++) {
            സ്ഥിരം uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            സ്ഥിരം dma_addr_t dst_row0 = dst->data  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            സ്ഥിരം dma_addr_t src_row0 = src0->data + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_തള്ളുക_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_തള്ളുക_range(q, dst_row0 + (size_t) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    തിരികെ HTP_STATUS_OK;
}

സ്ഥിതം ശൂന്യം roll_thread_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_roll_context * rctx = (ഘടന htp_roll_context *) data;
    ഘടന htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    സ്ഥിരം uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    എങ്കിൽ (row_start >= row_end) {
        തിരികെ;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    വേണ്ടി (uint32_t row = row_start; row < row_end; row++) {
        സ്ഥിരം uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        സ്ഥിരം uint32_t rem = row - i3 * ne2 * ne1;
        സ്ഥിരം uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        സ്ഥിരം uint32_t i1  = rem - i2 * ne1;

        സ്ഥിരം uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        സ്ഥിരം uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        സ്ഥിരം uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        സ്ഥിരം uint8_t * src_row = (സ്ഥിരം uint8_t *) (uintptr_t) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->data + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * വലുപ്പം(ദശാംശം), n0 * വലുപ്പം(ദശാംശം), ne0 * വലുപ്പം(ദശാംശം), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * വലുപ്പം(ദശാംശം), n0, വലുപ്പം(ദശാംശം));

        എങ്കിൽ (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * വലുപ്പം(ദശാംശം), ne0 * വലുപ്പം(ദശാംശം), 1);
            hvx_copy_uu(dst_row + n0 * വലുപ്പം(ദശാംശം), src_row, i0_src0, വലുപ്പം(ദശാംശം));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(മുകളിലെ_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

പൂർണ്ണസംഖ്യ execute_op_roll_f32(ഘടന htp_ops_context * octx) {
    htp_roll_preamble;

    എങ്കിൽ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (src0->nb[0] != വലുപ്പം(ദശാംശം) || dst->nb[0] != വലുപ്പം(ദശാംശം)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    സ്ഥിരം uint32_t total_rows = ne1 * ne2 * ne3;
    സ്ഥിരം size_t dst_row_size = ne0 * വലുപ്പം(ദശാംശം);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, വലുപ്പം(ദശാംശം), (uint32_t) dst_row_size, &rows_per_chunk);
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം <= 1) {
        എങ്കിൽ (htp_tensor_is_contiguous(src0, വലുപ്പം(ദശാംശം)) && htp_tensor_is_contiguous(dst, വലുപ്പം(ദശാംശം))) {
            തിരികെ roll_dma_f32_contiguous(octx);
        }
        തിരികെ roll_dma_f32_strided(octx);
    }

    എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    സ്ഥിരം uint32_t n_threads = octx->n_threads;
    ഘടന htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_മൂല്യങ്ങൾ(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_മൂല്യങ്ങൾ(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    തിരികെ HTP_STATUS_OK;
}

പൂർണ്ണസംഖ്യ op_roll(ഘടന htp_ops_context * octx) {
    തിരഞ്ഞെടുപ്പ് (octx->src[0]->type) {
        അവസ്ഥ HTP_TYPE_F32:
            തിരികെ execute_op_roll_f32(octx);

        സ്ഥിരസ്ഥിതി:
            തിരികെ HTP_STATUS_NO_SUPPORT;
    }
}
