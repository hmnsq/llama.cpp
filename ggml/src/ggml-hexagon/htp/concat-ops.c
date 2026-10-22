#அந்தர்பாவ "concat-ops.h"
#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "dma-copy.h"
#அந்தர்பாவ "hex-fastdiv.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "hexagon_protos.h"
#அந்தர்பாவ "hexagon_types.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-fence.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "htp-vtcm.h"
#அந்தர்பாவ "hvx-utils.h"
#அந்தர்பாவ "hvx_hexagon_protos.h"
#அந்தர்பாவ "gather-rows.h"

#அந்தர்பாவ <string.h>

கட்டமைப்பு htp_concat_context {
    கட்டமைப்பு htp_ops_context * octx;
    uint8_t * spad0_base;
    uint8_t * spad1_base;
    uint32_t  spad0_size_per_thread;
    uint32_t  spad1_size_per_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_per_thread;
    uint32_t  nplanes;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne2;
};

நிலையான உள்வரி dma_addr_t concat_plane_addr(மாறிலி கட்டமைப்பு htp_tensor * t, uint32_t p, மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div_ne2, uint32_t ne2) {
    மாறிலி uint32_t i3 = fastdiv(p, div_ne2);
    மாறிலி uint32_t i2 = p - i3 * ne2;
    திரும்பு t->data + i2 * t->nb[2] + i3 * t->nb[3];
}

நிலையான வெறுமை concat_2d_f32_transposed(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_concat_context * cctx = (கட்டமைப்பு htp_concat_context *) data;
    கட்டமைப்பு htp_ops_context * octx = cctx->octx;

    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    மாறிலி uint32_t src0_ne0 = src0->ne[0];
    மாறிலி uint32_t src1_ne0 = src1->ne[0];

    மாறிலி uint32_t row_end = cctx->row_start + cctx->nrows;
    மாறிலி uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    மாறிலி uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    எனில் (start_i >= end_i) திரும்பு;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    மாறிலி uint32_t block_i = 32;
    மாறிலி uint32_t spad1_stride = block_i * அளவு(தசம);

    மாறிலி HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    மாறிலி uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    மாறிலி uint32_t src0_row_bytes  = src0_ne0 * அளவு(தசம);
    மாறிலி uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    மாறிலி uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    மாறிலி uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * அளவு(தசம);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div_ne2 = &cctx->div_ne2;
    மாறிலி uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    மாறிலி dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_தள்ளு(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * அளவு(தசம), src1_ne0);

    மாறிலி dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_தள்ளு(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_எடு(dma_q); // src1

    போது (p < cctx->nplanes) {
        மாறிலி uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        க்கு (uint32_t j = 0; j < src1_ne0; j += 32) {
            மாறிலி uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * அளவு(தசம);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        எனில் (ni >= end_i) {
            ni = start_i;
            np++;
        }
        மாறிலி bool has_next = np < cctx->nplanes;
        மாறிலி uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        எனில் (has_next) {
            மாறிலி dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_தள்ளு(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * அளவு(தசம), src1_ne0);
        }

        dma_queue_எடு(dma_q); // src0

        மாறிலி dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * அளவு(தசம), current_block_i);

        எனில் (has_next) {
            dma_queue_எடு(dma_q); // next src1
        }
        dma_queue_எடு(dma_q); // dst

        // spad0 is free after the dst write
        எனில் (has_next) {
            மாறிலி dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_தள்ளு(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

நிலையான வெறுமை concat_2d_f16_transposed(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_concat_context * cctx = (கட்டமைப்பு htp_concat_context *) data;
    கட்டமைப்பு htp_ops_context * octx = cctx->octx;

    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    மாறிலி uint32_t src0_ne0 = src0->ne[0];
    மாறிலி uint32_t src1_ne0 = src1->ne[0];

    மாறிலி uint32_t row_end = cctx->row_start + cctx->nrows;
    மாறிலி uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    மாறிலி uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    எனில் (start_i >= end_i) திரும்பு;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    மாறிலி uint32_t block_i = 64;
    மாறிலி uint32_t spad1_stride = block_i * அளவு(__fp16);

    மாறிலி HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    மாறிலி uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    மாறிலி uint32_t src0_row_bytes  = src0_ne0 * அளவு(__fp16);
    மாறிலி uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    மாறிலி uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    மாறிலி uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * அளவு(__fp16);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div_ne2 = &cctx->div_ne2;
    மாறிலி uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    மாறிலி dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_தள்ளு(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * அளவு(__fp16), src1_ne0);

    மாறிலி dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_தள்ளு(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_எடு(dma_q); // src1

    போது (p < cctx->nplanes) {
        மாறிலி uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        க்கு (uint32_t j = 0; j < src1_ne0; j += 64) {
            மாறிலி uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * அளவு(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        எனில் (ni >= end_i) {
            ni = start_i;
            np++;
        }
        மாறிலி bool has_next = np < cctx->nplanes;
        மாறிலி uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        எனில் (has_next) {
            மாறிலி dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_தள்ளு(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * அளவு(__fp16), src1_ne0);
        }

        dma_queue_எடு(dma_q); // src0

        மாறிலி dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * அளவு(__fp16), current_block_i);

        எனில் (has_next) {
            dma_queue_எடு(dma_q); // next src1
        }
        dma_queue_எடு(dma_q); // dst

        // spad0 is free after the dst write
        எனில் (has_next) {
            மாறிலி dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_தள்ளு(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

நிலையான முழு concat_regular(கட்டமைப்பு htp_ops_context * octx, முழு dim, uint32_t type_size) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    கட்டமைப்பு htp_tensor view0 = *dst;
    கட்டமைப்பு htp_tensor view1 = *dst;
    க்கு (முழு d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.data += src0->ne[dim] * dst->nb[dim];

    மாறிலி uint32_t total_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = total_rows_0;
    uint32_t rstart1 = 0, nrows1 = total_rows_1;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            total_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        rstart0 = range0.start;
        nrows0  = range0.மொத்தம்;

        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        rstart1 = range1.start;
        nrows1  = range1.மொத்தம்;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_size, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_size, rstart1, nrows1);
    dma_queue_flush(q);

    திரும்பு HTP_STATUS_OK;
}

நிலையான முழு concat_transposed(கட்டமைப்பு htp_ops_context * octx, மாறிலி கட்டமைப்பு htp_concat_kernel_params * kparams, uint32_t type_size) {
    எனில் (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    மாறிலி கட்டமைப்பு htp_tensor * dst = octx->dst;

    மாறிலி uint32_t total_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;
    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
    }

    எனில் (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    எனில் (kparams->vtcm_size > octx->ctx->vtcm_size) {
        திரும்பு HTP_STATUS_VTCM_TOO_SMALL;
    }

    மாறிலி uint32_t n_threads = octx->n_threads;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    கட்டமைப்பு htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_base            = vtcm_base;
    cctx.spad1_base            = vtcm_base + n_threads * kparams->u.transposed.spad0_size_per_thread;
    cctx.spad0_size_per_thread = kparams->u.transposed.spad0_size_per_thread;
    cctx.spad1_size_per_thread = kparams->u.transposed.spad1_size_per_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_மதிப்புகள்(dst->ne[2]);
    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t worker_func = (type_size == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    திரும்பு HTP_STATUS_OK;
}

முழு op_concat(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_concat_kernel_params * kparams = (மாறிலி கட்டமைப்பு htp_concat_kernel_params *) octx->kernel_params;
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    முழு status = HTP_STATUS_OK;
    மாறிலி uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    தேர்வு (kparams->kernel_type) {
        நிலை HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams->dim, type_size);
            நிறுத்து;

        நிலை HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams, type_size);
            நிறுத்து;

        நிலை HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_threads);
            நிறுத்து;

        இயல்பு:
            status = HTP_STATUS_NO_SUPPORT;
            நிறுத்து;
    }

    htp_ops_context_set_status(octx, status);

    திரும்பு octx->status;
}
