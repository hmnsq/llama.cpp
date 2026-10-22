#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#समावेश <string.h>

#समावेश "dma-queue.h"
#समावेश "hvx-utils.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "hex-common.h"
#समावेश "hex-profile.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"

काष्ठा htp_roll_context {
    काष्ठा htp_ops_context * octx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_thread;

    काष्ठा fastdiv_मूल्यावली div_ne1;
    काष्ठा fastdiv_मूल्यावली div_ne2_ne1;
};

अटल अंतरभूत uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    अगर (i < 0) {
        वापस (uint32_t) (i + (int32_t) ne);
    }
    अगर ((uint32_t) i >= ne) {
        वापस (uint32_t) i - ne;
    }
    वापस (uint32_t) i;
}

#घोषणा htp_roll_preamble                             \
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];    \
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;       \
                                                      \
    स्थिर uint32_t ne0 = dst->ne[0];                  \
    स्थिर uint32_t ne1 = dst->ne[1];                  \
    स्थिर uint32_t ne2 = dst->ne[2];                  \
    स्थिर uint32_t ne3 = dst->ne[3];                  \
                                                      \
    स्थिर uint32_t nb01 = src0->nb[1];                \
    स्थिर uint32_t nb02 = src0->nb[2];                \
    स्थिर uint32_t nb03 = src0->nb[3];                \
                                                      \
    स्थिर uint32_t nb1 = dst->nb[1];                  \
    स्थिर uint32_t nb2 = dst->nb[2];                  \
    स्थिर uint32_t nb3 = dst->nb[3];                  \
                                                      \
    स्थिर int32_t s0 = octx->op_params[0];            \
    स्थिर int32_t s1 = octx->op_params[1];            \
    स्थिर int32_t s2 = octx->op_params[2];            \
    स्थिर int32_t s3 = octx->op_params[3];            \
                                                      \
    स्थिर uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    स्थिर uint32_t n0      = ne0 - i0_src0;

#घोषणा htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

अटल अंतरभूत व्योम roll_dma_धकेलो(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    अगर (bytes == 0 || nrows == 0) {
        वापस;
    }

    अगर (!dma_queue_धकेलो(q, dma_make_तथ्य(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_धकेलो(q, dma_make_तथ्य(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

अटल अंतरभूत व्योम roll_dma_धकेलो_rows(dma_queue *               q,
                                      स्थिर काष्ठा htp_tensor * dst,
                                      स्थिर काष्ठा htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_आकार,
                                      uint32_t                  i0_src0) {
    स्थिर dma_addr_t dst_base = dst->तथ्य + (माप_प्रकार) dst_row * row_आकार;
    स्थिर dma_addr_t src_base = src0->तथ्य + (माप_प्रकार) src_row * row_आकार;
    स्थिर uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_धकेलो(q, dst_base, src_base + (माप_प्रकार) i0_src0 * माप(भग्न),
                  row_आकार, row_आकार, n0 * माप(भग्न), nrows);
    roll_dma_धकेलो(q, dst_base + (माप_प्रकार) n0 * माप(भग्न), src_base,
                  row_आकार, row_आकार, i0_src0 * माप(भग्न), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
अटल अंतरभूत व्योम roll_dma_धकेलो_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_धकेलो(q, dst_row, src_row + (माप_प्रकार) i0_src0 * माप(भग्न),
                  dst_stride, src_stride, n0 * माप(भग्न), nrows);
    roll_dma_धकेलो(q, dst_row + (माप_प्रकार) n0 * माप(भग्न), src_row,
                  dst_stride, src_stride, i0_src0 * माप(भग्न), nrows);
}

अटल पूर्णांक roll_dma_f32_contiguous(काष्ठा htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    स्थिर uint32_t row_आकार = ne0 * माप(भग्न);

    अगर (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_धकेलो_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_आकार, i0_src0);
        dma_queue_flush(q);
        वापस HTP_STATUS_OK;
    }

    अगर (s1 == 0) {
        स्थिर uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        क्रम (uint32_t i3 = 0; i3 < ne3; i3++) {
            स्थिर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            स्थिर uint32_t dst_row0 = i3 * ne2 * ne1;
            स्थिर uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            स्थिर uint32_t n2_first = ne2 - i2_src0;

            roll_dma_धकेलो_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_आकार, i0_src0);
            roll_dma_धकेलो_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_आकार, i0_src0);
        }

        dma_queue_flush(q);
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    स्थिर uint32_t n1_first = ne1 - i1_src0;

    क्रम (uint32_t i3 = 0; i3 < ne3; i3++) {
        स्थिर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        क्रम (uint32_t i2 = 0; i2 < ne2; i2++) {
            स्थिर uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            स्थिर uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            स्थिर uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_धकेलो_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_आकार, i0_src0);
            roll_dma_धकेलो_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_आकार, i0_src0);
        }
    }

    dma_queue_flush(q);
    वापस HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
अटल पूर्णांक roll_dma_f32_strided(काष्ठा htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    स्थिर uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    स्थिर uint32_t n1_first = ne1 - i1_src0;

    क्रम (uint32_t i3 = 0; i3 < ne3; i3++) {
        स्थिर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        क्रम (uint32_t i2 = 0; i2 < ne2; i2++) {
            स्थिर uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            स्थिर dma_addr_t dst_row0 = dst->तथ्य  + (माप_प्रकार) i2  * nb2  + (माप_प्रकार) i3  * nb3;
            स्थिर dma_addr_t src_row0 = src0->तथ्य + (माप_प्रकार) i02 * nb02 + (माप_प्रकार) i03 * nb03;

            roll_dma_धकेलो_range(q, dst_row0, src_row0 + (माप_प्रकार) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0);
            roll_dma_धकेलो_range(q, dst_row0 + (माप_प्रकार) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_queue_flush(q);
    वापस HTP_STATUS_OK;
}

अटल व्योम roll_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_roll_context * rctx = (काष्ठा htp_roll_context *) तथ्य;
    काष्ठा htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    स्थिर uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    स्थिर uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->nrows);
    अगर (row_start >= row_end) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    क्रम (uint32_t row = row_start; row < row_end; row++) {
        स्थिर uint32_t i3  = fastdiv(row, &rctx->div_ne2_ne1);
        स्थिर uint32_t rem = row - i3 * ne2 * ne1;
        स्थिर uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        स्थिर uint32_t i1  = rem - i2 * ne1;

        स्थिर uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        स्थिर uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        स्थिर uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        स्थिर uint8_t * src_row = (स्थिर uint8_t *) (uintptr_t) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_row = (uint8_t *) (uintptr_t) dst->तथ्य + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_row + i0_src0 * माप(भग्न), n0 * माप(भग्न), ne0 * माप(भग्न), 1);
        hvx_copy_uu(dst_row, src_row + i0_src0 * माप(भग्न), n0, माप(भग्न));

        अगर (i0_src0 != 0) {
            hex_l2fetch(src_row, i0_src0 * माप(भग्न), ne0 * माप(भग्न), 1);
            hvx_copy_uu(dst_row + n0 * माप(भग्न), src_row, i0_src0, माप(भग्न));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(ऊपरी_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

पूर्णांक execute_op_roll_f32(काष्ठा htp_ops_context * octx) {
    htp_roll_preamble;

    अगर (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (src0->nb[0] != माप(भग्न) || dst->nb[0] != माप(भग्न)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर uint32_t योग_rows = ne1 * ne2 * ne3;
    स्थिर माप_प्रकार dst_row_आकार = ne0 * माप(भग्न);

    uint32_t row_start = 0;
    uint32_t nrows     = योग_rows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, माप(भग्न), (uint32_t) dst_row_आकार, &rows_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (octx->ctx->mdev.गिनती <= 1) {
        अगर (htp_tensor_is_contiguous(src0, माप(भग्न)) && htp_tensor_is_contiguous(dst, माप(भग्न))) {
            वापस roll_dma_f32_contiguous(octx);
        }
        वापस roll_dma_f32_strided(octx);
    }

    अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर uint32_t n_threads = octx->n_threads;
    काष्ठा htp_roll_context rctx = {
        .octx             = octx,
        .row_start        = row_start,
        .nrows            = nrows,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .div_ne1          = init_fastdiv_मूल्यावली(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_मूल्यावली(dst->ne[2] * dst->ne[1]),
    };

    work_queue_run(octx->ctx->work_queue, roll_thread_f32, &rctx, n_threads);

    वापस HTP_STATUS_OK;
}

पूर्णांक op_roll(काष्ठा htp_ops_context * octx) {
    चयन (octx->src[0]->type) {
        हाल HTP_TYPE_F32:
            वापस execute_op_roll_f32(octx);

        शेष:
            वापस HTP_STATUS_NO_SUPPORT;
    }
}
