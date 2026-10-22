#ಅನ್ತರ್ಭಾವ "concat-ops.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "dma-copy.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "hexagon_protos.h"
#ಅನ್ತರ್ಭಾವ "hexagon_types.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-fence.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx_hexagon_protos.h"
#ಅನ್ತರ್ಭಾವ "gather-rows.h"

#ಅನ್ತರ್ಭಾವ <string.h>

ಸಂರಚನಾ htp_concat_context {
    ಸಂರಚನಾ htp_ops_context * octx;
    uint8_t * spad0_base;
    uint8_t * spad1_base;
    uint32_t  spad0_size_per_thread;
    uint32_t  spad1_size_per_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_per_thread;
    uint32_t  nplanes;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು div_ne2;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ dma_addr_t concat_plane_addr(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t, uint32_t p, ಧ್ರುವ ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು * div_ne2, uint32_t ne2) {
    ಧ್ರುವ uint32_t i3 = fastdiv(p, div_ne2);
    ಧ್ರುವ uint32_t i2 = p - i3 * ne2;
    ಪ್ರತ್ಯಾಗಮ t->data + i2 * t->nb[2] + i3 * t->nb[3];
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_2d_f32_transposed(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_concat_context * cctx = (ಸಂರಚನಾ htp_concat_context *) data;
    ಸಂರಚನಾ htp_ops_context * octx = cctx->octx;

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಧ್ರುವ uint32_t src0_ne0 = src0->ne[0];
    ಧ್ರುವ uint32_t src1_ne0 = src1->ne[0];

    ಧ್ರುವ uint32_t row_end = cctx->row_start + cctx->nrows;
    ಧ್ರುವ uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    ಧ್ರುವ uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    ಯದಿ (start_i >= end_i) ಪ್ರತ್ಯಾಗಮ;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    ಧ್ರುವ uint32_t block_i = 32;
    ಧ್ರುವ uint32_t spad1_stride = block_i * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ಧ್ರುವ uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    ಧ್ರುವ uint32_t src0_row_bytes  = src0_ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    ಧ್ರುವ uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    ಧ್ರುವ uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಧ್ರುವ ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು * div_ne2 = &cctx->div_ne2;
    ಧ್ರುವ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ಧ್ರುವ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * ಪರಿಮಾಣ(ಭಿನ್ನ), src1_ne0);

    ಧ್ರುವ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_ತೆಗೆ(dma_q); // src1

    ಯಾವತ್ (p < cctx->nplanes) {
        ಧ್ರುವ uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ಪ್ರತಿ (uint32_t j = 0; j < src1_ne0; j += 32) {
            ಧ್ರುವ uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * ಪರಿಮಾಣ(ಭಿನ್ನ);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        ಯದಿ (ni >= end_i) {
            ni = start_i;
            np++;
        }
        ಧ್ರುವ bool has_next = np < cctx->nplanes;
        ಧ್ರುವ uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ಯದಿ (has_next) {
            ಧ್ರುವ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * ಪರಿಮಾಣ(ಭಿನ್ನ), src1_ne0);
        }

        dma_queue_ತೆಗೆ(dma_q); // src0

        ಧ್ರುವ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * ಪರಿಮಾಣ(ಭಿನ್ನ), current_block_i);

        ಯದಿ (has_next) {
            dma_queue_ತೆಗೆ(dma_q); // next src1
        }
        dma_queue_ತೆಗೆ(dma_q); // dst

        // spad0 is free after the dst write
        ಯದಿ (has_next) {
            ಧ್ರುವ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_2d_f16_transposed(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_concat_context * cctx = (ಸಂರಚನಾ htp_concat_context *) data;
    ಸಂರಚನಾ htp_ops_context * octx = cctx->octx;

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಧ್ರುವ uint32_t src0_ne0 = src0->ne[0];
    ಧ್ರುವ uint32_t src1_ne0 = src1->ne[0];

    ಧ್ರುವ uint32_t row_end = cctx->row_start + cctx->nrows;
    ಧ್ರುವ uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    ಧ್ರುವ uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    ಯದಿ (start_i >= end_i) ಪ್ರತ್ಯಾಗಮ;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    ಧ್ರುವ uint32_t block_i = 64;
    ಧ್ರುವ uint32_t spad1_stride = block_i * ಪರಿಮಾಣ(__fp16);

    ಧ್ರುವ HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ಧ್ರುವ uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    ಧ್ರುವ uint32_t src0_row_bytes  = src0_ne0 * ಪರಿಮಾಣ(__fp16);
    ಧ್ರುವ uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    ಧ್ರುವ uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    ಧ್ರುವ uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * ಪರಿಮಾಣ(__fp16);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಧ್ರುವ ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು * div_ne2 = &cctx->div_ne2;
    ಧ್ರುವ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    ಧ್ರುವ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * ಪರಿಮಾಣ(__fp16), src1_ne0);

    ಧ್ರುವ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_ತೆಗೆ(dma_q); // src1

    ಯಾವತ್ (p < cctx->nplanes) {
        ಧ್ರುವ uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ಪ್ರತಿ (uint32_t j = 0; j < src1_ne0; j += 64) {
            ಧ್ರುವ uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * ಪರಿಮಾಣ(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        ಯದಿ (ni >= end_i) {
            ni = start_i;
            np++;
        }
        ಧ್ರುವ bool has_next = np < cctx->nplanes;
        ಧ್ರುವ uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ಯದಿ (has_next) {
            ಧ್ರುವ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * ಪರಿಮಾಣ(__fp16), src1_ne0);
        }

        dma_queue_ತೆಗೆ(dma_q); // src0

        ಧ್ರುವ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * ಪರಿಮಾಣ(__fp16), current_block_i);

        ಯದಿ (has_next) {
            dma_queue_ತೆಗೆ(dma_q); // next src1
        }
        dma_queue_ತೆಗೆ(dma_q); // dst

        // spad0 is free after the dst write
        ಯದಿ (has_next) {
            ಧ್ರುವ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ concat_regular(ಸಂರಚನಾ htp_ops_context * octx, ಪೂರ್ಣಾಙ್ಕ dim, uint32_t type_size) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಸಂರಚನಾ htp_tensor view0 = *dst;
    ಸಂರಚನಾ htp_tensor view1 = *dst;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.data += src0->ne[dim] * dst->nb[dim];

    ಧ್ರುವ uint32_t total_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = total_rows_0;
    uint32_t rstart1 = 0, nrows1 = total_rows_1;

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            total_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        rstart0 = range0.start;
        nrows0  = range0.ಎಣಿಕೆ;

        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        rstart1 = range1.start;
        nrows1  = range1.ಎಣಿಕೆ;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_size, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_size, rstart1, nrows1);
    dma_queue_flush(q);

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ concat_transposed(ಸಂರಚನಾ htp_ops_context * octx, ಧ್ರುವ ಸಂರಚನಾ htp_concat_kernel_params * kparams, uint32_t type_size) {
    ಯದಿ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst = octx->dst;

    ಧ್ರುವ uint32_t total_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;
    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_start = range.start;
        nrows     = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಯದಿ (kparams->vtcm_size > octx->ctx->vtcm_size) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_VTCM_TOO_SMALL;
    }

    ಧ್ರುವ uint32_t n_threads = octx->n_threads;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    ಸಂರಚನಾ htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_base            = vtcm_base;
    cctx.spad1_base            = vtcm_base + n_threads * kparams->u.transposed.spad0_size_per_thread;
    cctx.spad0_size_per_thread = kparams->u.transposed.spad0_size_per_thread;
    cctx.spad1_size_per_thread = kparams->u.transposed.spad1_size_per_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[2]);
    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t worker_func = (type_size == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಪೂರ್ಣಾಙ್ಕ op_concat(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_concat_kernel_params * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_concat_kernel_params *) octx->kernel_params;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಪೂರ್ಣಾಙ್ಕ status = HTP_STATUS_OK;
    ಧ್ರುವ uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    ವಿಕಲ್ಪ (kparams->kernel_type) {
        ಪಕ್ಷ HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams->dim, type_size);
            ಭಙ್ಗ;

        ಪಕ್ಷ HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams, type_size);
            ಭಙ್ಗ;

        ಪಕ್ಷ HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_threads);
            ಭಙ್ಗ;

        ಸ್ವಭಾವ:
            status = HTP_STATUS_NO_SUPPORT;
            ಭಙ್ಗ;
    }

    htp_ops_context_set_status(octx, status);

    ಪ್ರತ್ಯಾಗಮ octx->status;
}
