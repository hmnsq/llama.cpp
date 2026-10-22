#অন্তর্ভাৱ "concat-ops.h"
#অন্তর্ভাৱ "dma-queue.h"
#অন্তর্ভাৱ "hex-common.h"
#অন্তর্ভাৱ "dma-copy.h"
#অন্তর্ভাৱ "hex-fastdiv.h"
#অন্তর্ভাৱ "hex-profile.h"
#অন্তর্ভাৱ "hexagon_protos.h"
#অন্তর্ভাৱ "hexagon_types.h"
#অন্তর্ভাৱ "htp-ctx.h"
#অন্তর্ভাৱ "htp-fence.h"
#অন্তর্ভাৱ "htp-ops.h"
#অন্তর্ভাৱ "htp-tensor.h"
#অন্তর্ভাৱ "htp-vtcm.h"
#অন্তর্ভাৱ "hvx-utils.h"
#অন্তর্ভাৱ "hvx_hexagon_protos.h"
#অন্তর্ভাৱ "gather-rows.h"

#অন্তর্ভাৱ <string.h>

সংরচনা htp_concat_context {
    সংরচনা htp_ops_context * octx;
    uint8_t * spad0_base;
    uint8_t * spad1_base;
    uint32_t  spad0_size_per_thread;
    uint32_t  spad1_size_per_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_per_thread;
    uint32_t  nplanes;
    সংরচনা fastdiv_মানসমূহ div_ne2;
};

স্থির অন্তঃস্থ dma_addr_t concat_plane_addr(ধ্রুৱ সংরচনা htp_tensor * t, uint32_t p, ধ্রুৱ সংরচনা fastdiv_মানসমূহ * div_ne2, uint32_t ne2) {
    ধ্রুৱ uint32_t i3 = fastdiv(p, div_ne2);
    ধ্রুৱ uint32_t i2 = p - i3 * ne2;
    ঘূৰাই t->data + i2 * t->nb[2] + i3 * t->nb[3];
}

স্থির শূন্য concat_2d_f32_transposed(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_concat_context * cctx = (সংরচনা htp_concat_context *) data;
    সংরচনা htp_ops_context * octx = cctx->octx;

    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    ধ্রুৱ uint32_t src0_ne0 = src0->ne[0];
    ধ্রুৱ uint32_t src1_ne0 = src1->ne[0];

    ধ্রুৱ uint32_t row_end = cctx->row_start + cctx->nrows;
    ধ্রুৱ uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    ধ্রুৱ uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    যদি (start_i >= end_i) ঘূৰাই;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    ধ্রুৱ uint32_t block_i = 32;
    ধ্রুৱ uint32_t spad1_stride = block_i * পরিমাণ(ভিন্ন);

    ধ্রুৱ HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ধ্রুৱ uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    ধ্রুৱ uint32_t src0_row_bytes  = src0_ne0 * পরিমাণ(ভিন্ন);
    ধ্রুৱ uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    ধ্রুৱ uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    ধ্রুৱ uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * পরিমাণ(ভিন্ন);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুৱ সংরচনা fastdiv_মানসমূহ * div_ne2 = &cctx->div_ne2;
    ধ্রুৱ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ধ্রুৱ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_ঠেলো(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * পরিমাণ(ভিন্ন), src1_ne0);

    ধ্রুৱ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_ঠেলো(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_বেৰ_কৰো(dma_q); // src1

    যেতিয়ালৈ (p < cctx->nplanes) {
        ধ্রুৱ uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        বাবে (uint32_t j = 0; j < src1_ne0; j += 32) {
            ধ্রুৱ uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * পরিমাণ(ভিন্ন);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        যদি (ni >= end_i) {
            ni = start_i;
            np++;
        }
        ধ্রুৱ bool has_next = np < cctx->nplanes;
        ধ্রুৱ uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        যদি (has_next) {
            ধ্রুৱ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_ঠেলো(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * পরিমাণ(ভিন্ন), src1_ne0);
        }

        dma_queue_বেৰ_কৰো(dma_q); // src0

        ধ্রুৱ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * পরিমাণ(ভিন্ন), current_block_i);

        যদি (has_next) {
            dma_queue_বেৰ_কৰো(dma_q); // next src1
        }
        dma_queue_বেৰ_কৰো(dma_q); // dst

        // spad0 is free after the dst write
        যদি (has_next) {
            ধ্রুৱ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_ঠেলো(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

স্থির শূন্য concat_2d_f16_transposed(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_concat_context * cctx = (সংরচনা htp_concat_context *) data;
    সংরচনা htp_ops_context * octx = cctx->octx;

    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    ধ্রুৱ uint32_t src0_ne0 = src0->ne[0];
    ধ্রুৱ uint32_t src1_ne0 = src1->ne[0];

    ধ্রুৱ uint32_t row_end = cctx->row_start + cctx->nrows;
    ধ্রুৱ uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    ধ্রুৱ uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    যদি (start_i >= end_i) ঘূৰাই;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    ধ্রুৱ uint32_t block_i = 64;
    ধ্রুৱ uint32_t spad1_stride = block_i * পরিমাণ(__fp16);

    ধ্রুৱ HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ধ্রুৱ uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    ধ্রুৱ uint32_t src0_row_bytes  = src0_ne0 * পরিমাণ(__fp16);
    ধ্রুৱ uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    ধ্রুৱ uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    ধ্রুৱ uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * পরিমাণ(__fp16);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুৱ সংরচনা fastdiv_মানসমূহ * div_ne2 = &cctx->div_ne2;
    ধ্রুৱ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ধ্রুৱ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_ঠেলো(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * পরিমাণ(__fp16), src1_ne0);

    ধ্রুৱ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_ঠেলো(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_বেৰ_কৰো(dma_q); // src1

    যেতিয়ালৈ (p < cctx->nplanes) {
        ধ্রুৱ uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        বাবে (uint32_t j = 0; j < src1_ne0; j += 64) {
            ধ্রুৱ uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * পরিমাণ(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        যদি (ni >= end_i) {
            ni = start_i;
            np++;
        }
        ধ্রুৱ bool has_next = np < cctx->nplanes;
        ধ্রুৱ uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        যদি (has_next) {
            ধ্রুৱ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_ঠেলো(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * পরিমাণ(__fp16), src1_ne0);
        }

        dma_queue_বেৰ_কৰো(dma_q); // src0

        ধ্রুৱ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * পরিমাণ(__fp16), current_block_i);

        যদি (has_next) {
            dma_queue_বেৰ_কৰো(dma_q); // next src1
        }
        dma_queue_বেৰ_কৰো(dma_q); // dst

        // spad0 is free after the dst write
        যদি (has_next) {
            ধ্রুৱ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_ঠেলো(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

স্থির পূর্ণাঙ্ক concat_regular(সংরচনা htp_ops_context * octx, পূর্ণাঙ্ক dim, uint32_t type_size) {
    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    সংরচনা htp_tensor view0 = *dst;
    সংরচনা htp_tensor view1 = *dst;
    বাবে (পূর্ণাঙ্ক d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.data += src0->ne[dim] * dst->nb[dim];

    ধ্রুৱ uint32_t total_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুৱ uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = total_rows_0;
    uint32_t rstart1 = 0, nrows1 = total_rows_1;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুৱ সংরচনা htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            total_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        rstart0 = range0.start;
        nrows0  = range0.গুনতি;

        ধ্রুৱ সংরচনা htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        rstart1 = range1.start;
        nrows1  = range1.গুনতি;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_size, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_size, rstart1, nrows1);
    dma_queue_flush(q);

    ঘূৰাই HTP_STATUS_OK;
}

স্থির পূর্ণাঙ্ক concat_transposed(সংরচনা htp_ops_context * octx, ধ্রুৱ সংরচনা htp_concat_kernel_params * kparams, uint32_t type_size) {
    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ঘূৰাই HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুৱ সংরচনা htp_tensor * dst = octx->dst;

    ধ্রুৱ uint32_t total_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুৱ সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        ঘূৰাই HTP_STATUS_OK;
    }

    যদি (kparams->vtcm_size > octx->ctx->vtcm_size) {
        ঘূৰাই HTP_STATUS_VTCM_TOO_SMALL;
    }

    ধ্রুৱ uint32_t n_threads = octx->n_threads;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    সংরচনা htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_base            = vtcm_base;
    cctx.spad1_base            = vtcm_base + n_threads * kparams->u.transposed.spad0_size_per_thread;
    cctx.spad0_size_per_thread = kparams->u.transposed.spad0_size_per_thread;
    cctx.spad1_size_per_thread = kparams->u.transposed.spad1_size_per_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_মানসমূহ(dst->ne[2]);
    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t worker_func = (type_size == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    ঘূৰাই HTP_STATUS_OK;
}

পূর্ণাঙ্ক op_concat(সংরচনা htp_ops_context * octx) {
    ধ্রুৱ সংরচনা htp_concat_kernel_params * kparams = (ধ্রুৱ সংরচনা htp_concat_kernel_params *) octx->kernel_params;
    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    পূর্ণাঙ্ক status = HTP_STATUS_OK;
    ধ্রুৱ uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    ৱিকল্প (kparams->kernel_type) {
        পক্ষ HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams->dim, type_size);
            ভঙ্গ;

        পক্ষ HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams, type_size);
            ভঙ্গ;

        পক্ষ HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_threads);
            ভঙ্গ;

        স্ৱভাৱ:
            status = HTP_STATUS_NO_SUPPORT;
            ভঙ্গ;
    }

    htp_ops_context_set_status(octx, status);

    ঘূৰাই octx->status;
}
