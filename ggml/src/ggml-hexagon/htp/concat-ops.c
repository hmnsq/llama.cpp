#ଅନ୍ତର୍ଭାଵ "concat-ops.h"
#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "dma-copy.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "hexagon_protos.h"
#ଅନ୍ତର୍ଭାଵ "hexagon_types.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-fence.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "htp-vtcm.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "hvx_hexagon_protos.h"
#ଅନ୍ତର୍ଭାଵ "gather-rows.h"

#ଅନ୍ତର୍ଭାଵ <string.h>

ସଂରଚନା htp_concat_context {
    ସଂରଚନା htp_ops_context * octx;
    uint8_t * spad0_base;
    uint8_t * spad1_base;
    uint32_t  spad0_size_per_thread;
    uint32_t  spad1_size_per_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_per_thread;
    uint32_t  nplanes;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne2;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_addr_t concat_plane_addr(ଅଚଳ ସଂରଚନା htp_tensor * t, uint32_t p, ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * div_ne2, uint32_t ne2) {
    ଅଚଳ uint32_t i3 = fastdiv(p, div_ne2);
    ଅଚଳ uint32_t i2 = p - i3 * ne2;
    ଫେରନ୍ତୁ t->data + i2 * t->nb[2] + i3 * t->nb[3];
}

ସ୍ଥିର ଶୂନ୍ୟ concat_2d_f32_transposed(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_concat_context * cctx = (ସଂରଚନା htp_concat_context *) data;
    ସଂରଚନା htp_ops_context * octx = cctx->octx;

    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଅଚଳ uint32_t src0_ne0 = src0->ne[0];
    ଅଚଳ uint32_t src1_ne0 = src1->ne[0];

    ଅଚଳ uint32_t row_end = cctx->row_start + cctx->nrows;
    ଅଚଳ uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    ଅଚଳ uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    ଯଦି (start_i >= end_i) ଫେରନ୍ତୁ;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    ଅଚଳ uint32_t block_i = 32;
    ଅଚଳ uint32_t spad1_stride = block_i * ଆକାର(ଦଶମିକ);

    ଅଚଳ HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ଅଚଳ uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    ଅଚଳ uint32_t src0_row_bytes  = src0_ne0 * ଆକାର(ଦଶମିକ);
    ଅଚଳ uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    ଅଚଳ uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    ଅଚଳ uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * ଆକାର(ଦଶମିକ);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * div_ne2 = &cctx->div_ne2;
    ଅଚଳ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ଅଚଳ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_ଠେଲ(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * ଆକାର(ଦଶମିକ), src1_ne0);

    ଅଚଳ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_ଠେଲ(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_କାଢ଼(dma_q); // src1

    ଯେତେବେଳେ (p < cctx->nplanes) {
        ଅଚଳ uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ପାଇଁ (uint32_t j = 0; j < src1_ne0; j += 32) {
            ଅଚଳ uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * ଆକାର(ଦଶମିକ);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        ଯଦି (ni >= end_i) {
            ni = start_i;
            np++;
        }
        ଅଚଳ bool has_next = np < cctx->nplanes;
        ଅଚଳ uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ଯଦି (has_next) {
            ଅଚଳ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_ଠେଲ(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * ଆକାର(ଦଶମିକ), src1_ne0);
        }

        dma_queue_କାଢ଼(dma_q); // src0

        ଅଚଳ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_ଠେଲ(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * ଆକାର(ଦଶମିକ), current_block_i);

        ଯଦି (has_next) {
            dma_queue_କାଢ଼(dma_q); // next src1
        }
        dma_queue_କାଢ଼(dma_q); // dst

        // spad0 is free after the dst write
        ଯଦି (has_next) {
            ଅଚଳ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_ଠେଲ(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

ସ୍ଥିର ଶୂନ୍ୟ concat_2d_f16_transposed(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_concat_context * cctx = (ସଂରଚନା htp_concat_context *) data;
    ସଂରଚନା htp_ops_context * octx = cctx->octx;

    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଅଚଳ uint32_t src0_ne0 = src0->ne[0];
    ଅଚଳ uint32_t src1_ne0 = src1->ne[0];

    ଅଚଳ uint32_t row_end = cctx->row_start + cctx->nrows;
    ଅଚଳ uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    ଅଚଳ uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    ଯଦି (start_i >= end_i) ଫେରନ୍ତୁ;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    ଅଚଳ uint32_t block_i = 64;
    ଅଚଳ uint32_t spad1_stride = block_i * ଆକାର(__fp16);

    ଅଚଳ HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ଅଚଳ uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    ଅଚଳ uint32_t src0_row_bytes  = src0_ne0 * ଆକାର(__fp16);
    ଅଚଳ uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    ଅଚଳ uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    ଅଚଳ uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * ଆକାର(__fp16);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * div_ne2 = &cctx->div_ne2;
    ଅଚଳ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ଅଚଳ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_ଠେଲ(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * ଆକାର(__fp16), src1_ne0);

    ଅଚଳ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_ଠେଲ(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_କାଢ଼(dma_q); // src1

    ଯେତେବେଳେ (p < cctx->nplanes) {
        ଅଚଳ uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ପାଇଁ (uint32_t j = 0; j < src1_ne0; j += 64) {
            ଅଚଳ uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * ଆକାର(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        ଯଦି (ni >= end_i) {
            ni = start_i;
            np++;
        }
        ଅଚଳ bool has_next = np < cctx->nplanes;
        ଅଚଳ uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ଯଦି (has_next) {
            ଅଚଳ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_ଠେଲ(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * ଆକାର(__fp16), src1_ne0);
        }

        dma_queue_କାଢ଼(dma_q); // src0

        ଅଚଳ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_ଠେଲ(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * ଆକାର(__fp16), current_block_i);

        ଯଦି (has_next) {
            dma_queue_କାଢ଼(dma_q); // next src1
        }
        dma_queue_କାଢ଼(dma_q); // dst

        // spad0 is free after the dst write
        ଯଦି (has_next) {
            ଅଚଳ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_ଠେଲ(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା concat_regular(ସଂରଚନା htp_ops_context * octx, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim, uint32_t type_size) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ସଂରଚନା htp_tensor view0 = *dst;
    ସଂରଚନା htp_tensor view1 = *dst;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.data += src0->ne[dim] * dst->nb[dim];

    ଅଚଳ uint32_t total_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = total_rows_0;
    uint32_t rstart1 = 0, nrows1 = total_rows_1;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            total_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        rstart0 = range0.start;
        nrows0  = range0.ଗଣନା_2;

        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        rstart1 = range1.start;
        nrows1  = range1.ଗଣନା_2;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_size, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_size, rstart1, nrows1);
    dma_queue_flush(q);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା concat_transposed(ସଂରଚନା htp_ops_context * octx, ଅଚଳ ସଂରଚନା htp_concat_kernel_params * kparams, uint32_t type_size) {
    ଯଦି (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst;

    ଅଚଳ uint32_t total_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;
    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଯଦି (kparams->vtcm_size > octx->ctx->vtcm_size) {
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    ସଂରଚନା htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_base            = vtcm_base;
    cctx.spad1_base            = vtcm_base + n_threads * kparams->u.transposed.spad0_size_per_thread;
    cctx.spad0_size_per_thread = kparams->u.transposed.spad0_size_per_thread;
    cctx.spad1_size_per_thread = kparams->u.transposed.spad1_size_per_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(dst->ne[2]);
    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t worker_func = (type_size == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_concat(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_concat_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_concat_kernel_params *) octx->kernel_params;
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = HTP_STATUS_OK;
    ଅଚଳ uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    ବାଛ (kparams->kernel_type) {
        ସ୍ଥିତି HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams->dim, type_size);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams, type_size);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_threads);
            ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            status = HTP_STATUS_NO_SUPPORT;
            ଭାଙ୍ଗ;
    }

    htp_ops_context_set_status(octx, status);

    ଫେରନ୍ତୁ octx->status;
}
