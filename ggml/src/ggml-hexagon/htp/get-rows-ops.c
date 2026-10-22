#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "hvx-quant.h"
#অন্তর্ভাব "get-rows-ops.h"
#অন্তর্ভাব "work-queue.h"

সংরচনা get_rows_context;

প্রকারনির্দেশ uint32_t (*get_rows_read_index_fn_t)(ধ্রুব শূন্য * src1, uint32_t idx, পূর্ণাঙ্ক idx_type);
প্রকারনির্দেশ শূন্য (*get_rows_flat_compute_fn_t)(ভিন্ন * dst_spad, ধ্রুব শূন্য * src_spad, uint32_t cur_elems);
প্রকারনির্দেশ শূন্য (*get_rows_tiled_compute_fn_t)(ভিন্ন * dst_spad, ধ্রুব uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row);

সংরচনা get_rows_context {
    সংরচনা htp_ops_context * octx;
    ধ্রুব সংরচনা htp_get_rows_kernel_params * kparams;
    সংরচনা htp_get_rows_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_base;
    ধ্রুব শূন্য * src1_data;
    get_rows_read_index_fn_t    read_index;
    get_rows_flat_compute_fn_t  compute_flat;
    get_rows_tiled_compute_fn_t compute_tiled;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
    uint32_t tile_size;
    uint32_t tile_stride;
    uint32_t src0_row_size;
    uint32_t dst_row_size;
    পূর্ণাঙ্ক idx_type;
};

#define get_rows_preamble                      \
    ধ্রুব uint32_t ne00 = octx->src[0]->ne[0]; \
    ধ্রুব uint32_t ne01 = octx->src[0]->ne[1]; \
    ধ্রুব uint32_t ne02 = octx->src[0]->ne[2]; \
    ধ্রুব uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ধ্রুব uint32_t ne10 = octx->src[1]->ne[0]; \
    ধ্রুব uint32_t ne11 = octx->src[1]->ne[1]; \
    ধ্রুব uint32_t ne12 = octx->src[1]->ne[2]; \
    ধ্রুব uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ধ্রুব uint32_t ne0 = octx->dst->ne[0];     \
    ধ্রুব uint32_t ne1 = octx->dst->ne[1];     \
    ধ্রুব uint32_t ne2 = octx->dst->ne[2];     \
    ধ্রুব uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ধ্রুব uint32_t nb01 = octx->src[0]->nb[1]; \
    ধ্রুব uint32_t nb02 = octx->src[0]->nb[2]; \
    ধ্রুব uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ধ্রুব uint32_t nb10 = octx->src[1]->nb[0]; \
    ধ্রুব uint32_t nb11 = octx->src[1]->nb[1]; \
    ধ্রুব uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ধ্রুব uint32_t nb1 = octx->dst->nb[1];     \
    ধ্রুব uint32_t nb2 = octx->dst->nb[2];     \
    ধ্রুব uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ধ্রুব uint32_t nr = ne10 * ne11 * ne12;

স্থির uint32_t get_rows_read_index_ddr(
    ধ্রুব শূন্য * src1,
    uint32_t idx,
    পূর্ণাঙ্ক idx_type
) {
    ধ্রুব uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? পরিমাণ(int64_t) : পরিমাণ(int32_t));
    প্রত্যাগম *(ধ্রুব uint32_t *) ((ধ্রুব uint8_t *) src1 + off);
}

স্থির __attribute__((noinline)) uint32_t get_rows_read_index_vtcm(
    ধ্রুব শূন্য * src1,
    uint32_t idx,
    পূর্ণাঙ্ক idx_type
) {
    ধ্রুব uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? পরিমাণ(int64_t) : পরিমাণ(int32_t));
    ধ্রুব HVX_Vector v = hvx_vmem((ধ্রুব uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    প্রত্যাগম (uint32_t) Q6_R_vextract_VR(v, off);
}

স্থির শূন্য get_rows_thread_sametype(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা get_rows_context * grctx = (সংরচনা get_rows_context *) data;
    সংরচনা htp_ops_context * octx = grctx->octx;
    ধ্রুব সংরচনা htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ধ্রুব uint32_t dr  = grctx->tasks_per_thread;
    ধ্রুব uint32_t ir0 = grctx->task_start + dr * ith;
    যদি (ir0 >= grctx->task_start + grctx->tasks) {
        প্রত্যাগম;
    }
    ধ্রুব uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    ধ্রুব uint32_t row_size_bytes = grctx->src0_row_size;
    dma_queue * dma_q = octx->ctx->dma[ith];

    প্রতি (uint32_t i = ir0; i < ir1; ++i) {
        ধ্রুব uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ধ্রুব uint32_t rem = i - i12 * ne11 * ne10;
        ধ্রুব uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ধ্রুব uint32_t i10 = rem - i11 * ne10;

        ধ্রুব uint32_t i01 = grctx->read_index(grctx->src1_data, i, grctx->idx_type);
        প্রতিজ্ঞা(i01 < ne01);

        ধ্রুব uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ধ্রুব uint32_t i02 = i11 - q02 * ne02;
        ধ্রুব uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ধ্রুব uint32_t i03 = i12 - q03 * ne03;

        ধ্রুব dma_addr_t src0_data = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;
        ধ্রুব dma_addr_t dst_data  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3;

        যাবত্ (!dma_queue_ঠেলো(dma_q, dma_make_data(dst_data, src0_data), nb1, nb01, row_size_bytes, 1)) {
            dma_queue_বের_করো(dma_q);
        }
    }
    dma_queue_flush(dma_q);
}



সংরচনা get_rows_flat_task {
    dma_addr_t src0_data;
    dma_addr_t dst_data;
};

স্থির অন্তঃস্থ সংরচনা get_rows_flat_task get_rows_flat_calc_task(
    ধ্রুব সংরচনা htp_ops_context * octx,
    ধ্রুব সংরচনা get_rows_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ধ্রুব সংরচনা htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ধ্রুব uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ধ্রুব uint32_t rem = i - i12 * ne11 * ne10;
    ধ্রুব uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ধ্রুব uint32_t i10 = rem - i11 * ne10;

    সংরচনা get_rows_flat_task task;
    task.src0_data = 0;
    task.dst_data  = octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3;

    যদি (need_src0) {
        ধ্রুব uint32_t i01 = grctx->read_index(grctx->src1_data, i, grctx->idx_type);
        প্রতিজ্ঞা(i01 < ne01);

        ধ্রুব uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ধ্রুব uint32_t i02 = i11 - q02 * ne02;
        ধ্রুব uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ধ্রুব uint32_t i03 = i12 - q03 * ne03;

        task.src0_data = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;
    }

    প্রত্যাগম task;
}

স্থির শূন্য get_rows_thread_flat(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা get_rows_context * grctx = (সংরচনা get_rows_context *) data;
    সংরচনা htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    ধ্রুব uint32_t dr  = grctx->tasks_per_thread;
    ধ্রুব uint32_t ir0 = grctx->task_start + dr * ith;
    যদি (ir0 >= grctx->task_start + grctx->tasks) {
        প্রত্যাগম;
    }
    ধ্রুব uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);

    dma_queue * dma_q = octx->ctx->dma[ith];
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    ধ্রুব সংরচনা htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    ধ্রুব uint32_t src0_bytes = grctx->src0_row_size;
    ধ্রুব uint32_t dst_bytes  = grctx->dst_row_size;
    get_rows_flat_compute_fn_t compute = grctx->compute_flat;

    প্রতি (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, spad_idx++) {
        ধ্রুব uint32_t i = ir0 + step;
        সংরচনা get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, true);

        dma_queue_ঠেলো(dma_q,
                       dma_make_data(octx->dst->data,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 0);
        dma_queue_ঠেলো(dma_q,
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size, task.src0_data),
                       vtcm_layout->src0_spad_half_size, src0_bytes, src0_bytes, 1);
    }

    প্রতি (uint32_t step = 0; step < ir1 - ir0; ++step) {
        ধ্রুব uint32_t i = ir0 + step;
        শূন্য * dst_spad = (শূন্য *) dma_queue_বের_করো(dma_q).src;
        শূন্য * src_spad = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

        সংরচনা get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((ভিন্ন *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_queue_ঠেলো(dma_q,
                       dma_make_data(task.dst_data, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 1);

        ধ্রুব uint32_t next_step = step + 2;
        যদি (next_step < ir1 - ir0) {
            ধ্রুব uint32_t pi = ir0 + next_step;
            সংরচনা get_rows_flat_task next_task = get_rows_flat_calc_task(octx, grctx, pi, true);
            dma_queue_ঠেলো(dma_q,
                           dma_make_data(src_spad, next_task.src0_data),
                           vtcm_layout->src0_spad_half_size, src0_bytes, src0_bytes, 1);
        }
    }

    dma_queue_flush(dma_q);
}



সংরচনা get_rows_tiled_task {
    dma_addr_t tile_src_base;
    dma_addr_t dst_data;
    uint32_t   row;
};

স্থির অন্তঃস্থ সংরচনা get_rows_tiled_task get_rows_tiled_calc_task(
    ধ্রুব সংরচনা htp_ops_context * octx,
    ধ্রুব সংরচনা get_rows_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_size
) {
    ধ্রুব সংরচনা htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ধ্রুব uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ধ্রুব uint32_t rem = i - i12 * ne11 * ne10;
    ধ্রুব uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ধ্রুব uint32_t i10 = rem - i11 * ne10;
    ধ্রুব uint32_t i01 = grctx->read_index(grctx->src1_data, i, grctx->idx_type);
    প্রতিজ্ঞা(i01 < ne01);

    ধ্রুব uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ধ্রুব uint32_t i02 = i11 - q02 * ne02;
    ধ্রুব uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ধ্রুব uint32_t i03 = i12 - q03 * ne03;
    ধ্রুব uint32_t column_tile = i01 / HTP_TENSOR_TILE_N_ROWS;
    ধ্রুব uint32_t row = i01 % HTP_TENSOR_TILE_N_ROWS;
    ধ্রুব dma_addr_t matrix = octx->src[0]->data + i02*nb02 + i03*nb03;

    সংরচনা get_rows_tiled_task task;
    task.tile_src_base = matrix + (column_tile * n_k_tiles) * tile_size;
    task.dst_data      = octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3;
    task.row           = row;
    প্রত্যাগম task;
}

স্থির শূন্য get_rows_thread_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা get_rows_context * grctx = (সংরচনা get_rows_context *) data;
    সংরচনা htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    ধ্রুব uint32_t dr  = grctx->tasks_per_thread;
    ধ্রুব uint32_t ir0 = grctx->task_start + dr * ith;
    যদি (ir0 >= grctx->task_start + grctx->tasks) {
        প্রত্যাগম;
    }

    ধ্রুব uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    ধ্রুব uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ধ্রুব সংরচনা htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * dst_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    dma_queue * dma_q = octx->ctx->dma[ith];
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব uint32_t tile_size   = grctx->tile_size;
    ধ্রুব uint32_t tile_stride = grctx->tile_stride;
    ধ্রুব uint32_t dst_bytes   = grctx->dst_row_size;
    get_rows_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    প্রতি (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, ++spad_idx) {
        ধ্রুব uint32_t i = ir0 + step;
        সংরচনা get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_size);

        // Dummy writeback to prime the queue with dst descriptor
        dma_queue_ঠেলো(dma_q,
                       dma_make_data(task.dst_data, dst_spad_base + spad_idx * vtcm_layout->dst_spad_half_size),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 0);

        // Prefetch row tiles
        dma_queue_ঠেলো(dma_q,
                       dma_make_data(src_spad_base + spad_idx * vtcm_layout->src0_spad_half_size, task.tile_src_base),
                       tile_stride, tile_size, tile_size, n_k_tiles);
    }

    প্রতি (uint32_t step = 0; step < ir1 - ir0; ++step) {
        ধ্রুব uint32_t i = ir0 + step;
        ভিন্ন * dst_spad   = (ভিন্ন *) dma_queue_বের_করো(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        সংরচনা get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_size);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.row);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_queue_ঠেলো(dma_q,
                       dma_make_data(task.dst_data, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 1);

        ধ্রুব uint32_t next_step = step + 2;
        যদি (next_step < ir1 - ir0) {
            ধ্রুব uint32_t ni = ir0 + next_step;
            সংরচনা get_rows_tiled_task next_task = get_rows_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_size);
            dma_queue_ঠেলো(dma_q,
                           dma_make_data(src_spad, next_task.tile_src_base),
                           tile_stride, tile_size, tile_size, n_k_tiles);
        }
    }

    dma_queue_flush(dma_q);
}

স্থির __attribute__((noinline)) শূন্য get_rows_dma_src1(
    সংরচনা htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_size
) {
    ধ্রুব uint32_t ne10 = octx->src[1]->ne[0];
    ধ্রুব uint32_t ne11 = octx->src[1]->ne[1];
    ধ্রুব uint32_t ne12 = octx->src[1]->ne[2];
    ধ্রুব uint32_t ne13 = octx->src[1]->ne[3];
    ধ্রুব uint32_t nb11 = octx->src[1]->nb[1];
    ধ্রুব uint32_t nb12 = octx->src[1]->nb[2];
    ধ্রুব uint32_t nb13 = octx->src[1]->nb[3];

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    প্রতি (uint32_t i13 = 0; i13 < ne13; ++i13) {
        প্রতি (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ধ্রুব dma_addr_t src_plane = octx->src[1]->data + i12 * nb12 + i13 * nb13;
            যাবত্ (!dma_queue_ঠেলো(dma_q, dma_make_data(cur_dst, src_plane),
                                   ne10 * elem_size, nb11, ne10 * elem_size, ne11)) {
                dma_queue_বের_করো(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_size;
        }
    }
    dma_queue_flush(dma_q);
}

পূর্ণাঙ্ক op_get_rows(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_get_rows_kernel_params * kparams = (ধ্রুব সংরচনা htp_get_rows_kernel_params *) octx->kernel_params;

    ধ্রুব uint32_t ne00          = octx->src[0]->ne[0];
    ধ্রুব uint32_t src0_row_size = htp_tensor_get_row_size(octx->src[0]->type, ne00);
    যদি (src0_row_size == 0) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    যদি (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) {
        যদি (octx->src[0]->type != octx->dst->type) {
            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
        }
    } অন্যথা {
        যদি (octx->dst->type != HTP_TYPE_F32) {
            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
        }
    }

    যদি (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব সংরচনা htp_tensor * dst = octx->dst;
    ধ্রুব uint32_t total_tasks    = kparams->total_tasks;
    ধ্রুব uint32_t dst_row_size   = (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) ? src0_row_size : ne00 * পরিমাণ(ভিন্ন);

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t tasks_per_chunk = 1;
        htp_tensor_mdev_rows_per_chunk(dst, dst_row_size / dst->ne[0], dst_row_size, &tasks_per_chunk);
        ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, tasks_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        task_start = range.start;
        tasks      = range.গুনতি;
    }

    যদি (tasks == 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }

    যদি (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুব uint32_t n_threads = octx->n_threads;

    সংরচনা get_rows_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;
    grctx.compute_flat = রিক্ত;
    grctx.compute_tiled = রিক্ত;
    grctx.task_start = task_start;
    grctx.tasks = tasks;
    grctx.tasks_per_thread = octx->ctx->mdev.গুনতি == 1 ? kparams->tasks_per_thread : fastdiv(tasks + n_threads - 1, &octx->n_threads_div);
    grctx.tile_size   = 0;
    grctx.tile_stride = 0;
    grctx.src0_row_size = src0_row_size;
    grctx.dst_row_size  = dst_row_size;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_rows_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, total_tasks, grctx.idx_type, n_threads);

    যদি (grctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, grctx.vtcm_layout.total_bytes);
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }

    work_queue_func_t q_func = রিক্ত;
    বিকল্প (kparams->kernel_type) {
        পক্ষ HTP_GET_ROWS_KERNEL_SAMETYPE:
            q_func = get_rows_thread_sametype;
            ভঙ্গ;
        পক্ষ HTP_GET_ROWS_KERNEL_TILED:
            বিকল্প (octx->src[0]->type) {
                পক্ষ HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_row_tiled_f16_f32;  ভঙ্গ;
                পক্ষ HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q1_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q2_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q8_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_k_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q5_k_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q6_k_f32; ভঙ্গ;
                স্বভাব:            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
            }
            grctx.tile_size   = htp_tensor_get_tile_size(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_size(octx->src[0]->type);
            q_func = get_rows_thread_tiled;
            ভঙ্গ;
        পক্ষ HTP_GET_ROWS_KERNEL_FLAT:
            বিকল্প (octx->src[0]->type) {
                পক্ষ HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_row_flat_f16_f32;  ভঙ্গ;
                পক্ষ HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_row_flat_bf16_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_row_flat_q1_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_row_flat_q2_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_row_flat_q8_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_row_flat_q4_0_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_row_flat_q4_k_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_row_flat_q5_k_f32; ভঙ্গ;
                পক্ষ HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_row_flat_q6_k_f32; ভঙ্গ;
                স্বভাব:            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
            }
            q_func = get_rows_thread_flat;
            ভঙ্গ;
        স্বভাব:
            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব uint32_t elem_size = (grctx.idx_type == HTP_TYPE_I64) ? পরিমাণ(int64_t) : পরিমাণ(int32_t);
    ধ্রুব size_t src1_bytes  = (size_t) total_tasks * elem_size;

    যদি (!htp_tensor_is_extended(octx->src[1]) && src1_bytes <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_size)) {
        grctx.src1_data  = (ধ্রুব শূন্য *) (uintptr_t) octx->src[1]->data;
        grctx.read_index = get_rows_read_index_ddr;
    } অন্যথা {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_base, grctx.vtcm_layout.off_src1);
        get_rows_dma_src1(octx, vtcm_src1, elem_size);
        grctx.src1_data  = vtcm_src1;
        grctx.read_index = get_rows_read_index_vtcm;
    }

    FARF(উপরের_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         grctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         kparams->kernel_type, n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &grctx, n_threads);
    প্রত্যাগম HTP_STATUS_OK;
}
