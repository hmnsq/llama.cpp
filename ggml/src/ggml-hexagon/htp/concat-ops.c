#समावेश "concat-ops.h"
#समावेश "dma-queue.h"
#समावेश "hex-common.h"
#समावेश "dma-copy.h"
#समावेश "hex-fastdiv.h"
#समावेश "hex-profile.h"
#समावेश "hexagon_protos.h"
#समावेश "hexagon_types.h"
#समावेश "htp-ctx.h"
#समावेश "htp-fence.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "htp-vtcm.h"
#समावेश "hvx-utils.h"
#समावेश "hvx_hexagon_protos.h"
#समावेश "gather-rows.h"

#समावेश <string.h>

काष्ठा htp_concat_context {
    काष्ठा htp_ops_context * octx;
    uint8_t * spad0_base;
    uint8_t * spad1_base;
    uint32_t  spad0_आकार_per_thread;
    uint32_t  spad1_आकार_per_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_per_thread;
    uint32_t  nplanes;
    काष्ठा fastdiv_मूल्यावली div_ne2;
};

अटल अंतरभूत dma_addr_t concat_plane_addr(स्थिर काष्ठा htp_tensor * t, uint32_t p, स्थिर काष्ठा fastdiv_मूल्यावली * div_ne2, uint32_t ne2) {
    स्थिर uint32_t i3 = fastdiv(p, div_ne2);
    स्थिर uint32_t i2 = p - i3 * ne2;
    वापस t->तथ्य + i2 * t->nb[2] + i3 * t->nb[3];
}

अटल व्योम concat_2d_f32_transposed(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_concat_context * cctx = (काष्ठा htp_concat_context *) तथ्य;
    काष्ठा htp_ops_context * octx = cctx->octx;

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर uint32_t src0_ne0 = src0->ne[0];
    स्थिर uint32_t src1_ne0 = src1->ne[0];

    स्थिर uint32_t row_end = cctx->row_start + cctx->nrows;
    स्थिर uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    स्थिर uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    अगर (start_i >= end_i) वापस;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_आकार_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_आकार_per_thread;

    स्थिर uint32_t block_i = 32;
    स्थिर uint32_t spad1_stride = block_i * माप(भग्न);

    स्थिर HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    स्थिर uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    स्थिर uint32_t src0_row_bytes  = src0_ne0 * माप(भग्न);
    स्थिर uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    स्थिर uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    स्थिर uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * माप(भग्न);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne2 = &cctx->div_ne2;
    स्थिर uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    स्थिर dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * माप(भग्न), src1_ne0);

    स्थिर dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_निकालो(dma_q); // src1

    जबतक (p < cctx->nplanes) {
        स्थिर uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        क्रम (uint32_t j = 0; j < src1_ne0; j += 32) {
            स्थिर uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * माप(भग्न);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        अगर (ni >= end_i) {
            ni = start_i;
            np++;
        }
        स्थिर bool has_अग्र = np < cctx->nplanes;
        स्थिर uint32_t अग्र_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        अगर (has_अग्र) {
            स्थिर dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], अग्र_block_i * माप(भग्न), src1_ne0);
        }

        dma_queue_निकालो(dma_q); // src0

        स्थिर dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * माप(भग्न), current_block_i);

        अगर (has_अग्र) {
            dma_queue_निकालो(dma_q); // next src1
        }
        dma_queue_निकालो(dma_q); // dst

        // spad0 is free after the dst write
        अगर (has_अग्र) {
            स्थिर dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, अग्र_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

अटल व्योम concat_2d_f16_transposed(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_concat_context * cctx = (काष्ठा htp_concat_context *) तथ्य;
    काष्ठा htp_ops_context * octx = cctx->octx;

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर uint32_t src0_ne0 = src0->ne[0];
    स्थिर uint32_t src1_ne0 = src1->ne[0];

    स्थिर uint32_t row_end = cctx->row_start + cctx->nrows;
    स्थिर uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    स्थिर uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    अगर (start_i >= end_i) वापस;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_आकार_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_आकार_per_thread;

    स्थिर uint32_t block_i = 64;
    स्थिर uint32_t spad1_stride = block_i * माप(__fp16);

    स्थिर HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    स्थिर uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    स्थिर uint32_t src0_row_bytes  = src0_ne0 * माप(__fp16);
    स्थिर uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    स्थिर uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    स्थिर uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * माप(__fp16);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne2 = &cctx->div_ne2;
    स्थिर uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    स्थिर dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * माप(__fp16), src1_ne0);

    स्थिर dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_निकालो(dma_q); // src1

    जबतक (p < cctx->nplanes) {
        स्थिर uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        क्रम (uint32_t j = 0; j < src1_ne0; j += 64) {
            स्थिर uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * माप(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        अगर (ni >= end_i) {
            ni = start_i;
            np++;
        }
        स्थिर bool has_अग्र = np < cctx->nplanes;
        स्थिर uint32_t अग्र_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        अगर (has_अग्र) {
            स्थिर dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], अग्र_block_i * माप(__fp16), src1_ne0);
        }

        dma_queue_निकालो(dma_q); // src0

        स्थिर dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * माप(__fp16), current_block_i);

        अगर (has_अग्र) {
            dma_queue_निकालो(dma_q); // next src1
        }
        dma_queue_निकालो(dma_q); // dst

        // spad0 is free after the dst write
        अगर (has_अग्र) {
            स्थिर dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, अग्र_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

अटल पूर्णांक concat_regular(काष्ठा htp_ops_context * octx, पूर्णांक dim, uint32_t type_आकार) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    काष्ठा htp_tensor view0 = *dst;
    काष्ठा htp_tensor view1 = *dst;
    क्रम (पूर्णांक d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.तथ्य += src0->ne[dim] * dst->nb[dim];

    स्थिर uint32_t योग_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर uint32_t योग_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = योग_rows_0;
    uint32_t rstart1 = 0, nrows1 = योग_rows_1;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर काष्ठा htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            योग_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        rstart0 = range0.start;
        nrows0  = range0.गिनती;

        स्थिर काष्ठा htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            योग_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        rstart1 = range1.start;
        nrows1  = range1.गिनती;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_आकार, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_आकार, rstart1, nrows1);
    dma_queue_flush(q);

    वापस HTP_STATUS_OK;
}

अटल पूर्णांक concat_transposed(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_concat_kernel_params * kparams, uint32_t type_आकार) {
    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर काष्ठा htp_tensor * dst = octx->dst;

    स्थिर uint32_t योग_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = योग_rows;
    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (kparams->vtcm_आकार > octx->ctx->vtcm_आकार) {
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    स्थिर uint32_t n_threads = octx->n_threads;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    काष्ठा htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_base            = vtcm_base;
    cctx.spad1_base            = vtcm_base + n_threads * kparams->u.transposed.spad0_आकार_per_thread;
    cctx.spad0_आकार_per_thread = kparams->u.transposed.spad0_आकार_per_thread;
    cctx.spad1_आकार_per_thread = kparams->u.transposed.spad1_आकार_per_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_मूल्यावली(dst->ne[2]);
    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t worker_func = (type_आकार == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    वापस HTP_STATUS_OK;
}

पूर्णांक op_concat(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_concat_kernel_params * kparams = (स्थिर काष्ठा htp_concat_kernel_params *) octx->kernel_params;
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    पूर्णांक status = HTP_STATUS_OK;
    स्थिर uint32_t type_आकार = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    चयन (kparams->kernel_type) {
        हाल HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams->dim, type_आकार);
            अवरोध;

        हाल HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams, type_आकार);
            अवरोध;

        हाल HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_threads);
            अवरोध;

        शेष:
            status = HTP_STATUS_NO_SUPPORT;
            अवरोध;
    }

    htp_ops_context_set_status(octx, status);

    वापस octx->status;
}
