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

কাঠামো get_rows_context;

প্রকারলেখ uint32_t (*get_rows_read_সূচক_fn_t)(ধ্রুবক শূন্য * src1, uint32_t idx, পূর্ণসংখ্যা idx_type);
প্রকারলেখ শূন্য (*get_rows_flat_compute_fn_t)(দশমিক * dst_spad, ধ্রুবক শূন্য * src_spad, uint32_t cur_elems);
প্রকারলেখ শূন্য (*get_rows_tiled_compute_fn_t)(দশমিক * dst_spad, ধ্রুবক uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row);

কাঠামো get_rows_context {
    কাঠামো htp_ops_context * octx;
    ধ্রুবক কাঠামো htp_get_rows_kernel_params * kparams;
    কাঠামো htp_get_rows_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_base;
    ধ্রুবক শূন্য * src1_তথ্য;
    get_rows_read_সূচক_fn_t    read_সূচক;
    get_rows_flat_compute_fn_t  compute_flat;
    get_rows_tiled_compute_fn_t compute_tiled;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
    uint32_t tile_আকার;
    uint32_t tile_stride;
    uint32_t src0_row_আকার;
    uint32_t dst_row_আকার;
    পূর্ণসংখ্যা idx_type;
};

#define get_rows_preamble                      \
    ধ্রুবক uint32_t ne00 = octx->src[0]->ne[0]; \
    ধ্রুবক uint32_t ne01 = octx->src[0]->ne[1]; \
    ধ্রুবক uint32_t ne02 = octx->src[0]->ne[2]; \
    ধ্রুবক uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ধ্রুবক uint32_t ne10 = octx->src[1]->ne[0]; \
    ধ্রুবক uint32_t ne11 = octx->src[1]->ne[1]; \
    ধ্রুবক uint32_t ne12 = octx->src[1]->ne[2]; \
    ধ্রুবক uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ধ্রুবক uint32_t ne0 = octx->dst->ne[0];     \
    ধ্রুবক uint32_t ne1 = octx->dst->ne[1];     \
    ধ্রুবক uint32_t ne2 = octx->dst->ne[2];     \
    ধ্রুবক uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ধ্রুবক uint32_t nb01 = octx->src[0]->nb[1]; \
    ধ্রুবক uint32_t nb02 = octx->src[0]->nb[2]; \
    ধ্রুবক uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ধ্রুবক uint32_t nb10 = octx->src[1]->nb[0]; \
    ধ্রুবক uint32_t nb11 = octx->src[1]->nb[1]; \
    ধ্রুবক uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ধ্রুবক uint32_t nb1 = octx->dst->nb[1];     \
    ধ্রুবক uint32_t nb2 = octx->dst->nb[2];     \
    ধ্রুবক uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ধ্রুবক uint32_t nr = ne10 * ne11 * ne12;

স্থির uint32_t get_rows_read_সূচক_ddr(
    ধ্রুবক শূন্য * src1,
    uint32_t idx,
    পূর্ণসংখ্যা idx_type
) {
    ধ্রুবক uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? আকার(int64_t) : আকার(int32_t));
    ফেরত *(ধ্রুবক uint32_t *) ((ধ্রুবক uint8_t *) src1 + off);
}

স্থির __attribute__((noinline)) uint32_t get_rows_read_সূচক_vtcm(
    ধ্রুবক শূন্য * src1,
    uint32_t idx,
    পূর্ণসংখ্যা idx_type
) {
    ধ্রুবক uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? আকার(int64_t) : আকার(int32_t));
    ধ্রুবক HVX_Vector v = hvx_vmem((ধ্রুবক uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    ফেরত (uint32_t) Q6_R_vextract_VR(v, off);
}

স্থির শূন্য get_rows_thread_sametype(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো get_rows_context * grctx = (কাঠামো get_rows_context *) তথ্য;
    কাঠামো htp_ops_context * octx = grctx->octx;
    ধ্রুবক কাঠামো htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ধ্রুবক uint32_t dr  = grctx->tasks_per_thread;
    ধ্রুবক uint32_t ir0 = grctx->task_start + dr * ith;
    যদি (ir0 >= grctx->task_start + grctx->tasks) {
        ফেরত;
    }
    ধ্রুবক uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    ধ্রুবক uint32_t row_আকার_bytes = grctx->src0_row_আকার;
    dma_queue * dma_q = octx->ctx->dma[ith];

    জন্য (uint32_t i = ir0; i < ir1; ++i) {
        ধ্রুবক uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ধ্রুবক uint32_t rem = i - i12 * ne11 * ne10;
        ধ্রুবক uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ধ্রুবক uint32_t i10 = rem - i11 * ne10;

        ধ্রুবক uint32_t i01 = grctx->read_সূচক(grctx->src1_তথ্য, i, grctx->idx_type);
        প্রতিজ্ঞা(i01 < ne01);

        ধ্রুবক uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ধ্রুবক uint32_t i02 = i11 - q02 * ne02;
        ধ্রুবক uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ধ্রুবক uint32_t i03 = i12 - q03 * ne03;

        ধ্রুবক dma_addr_t src0_তথ্য = octx->src[0]->তথ্য + i01*nb01 + i02*nb02 + i03*nb03;
        ধ্রুবক dma_addr_t dst_তথ্য  = octx->dst->তথ্য    + i10*nb1  + i11*nb2  + i12*nb3;

        যখন (!dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_তথ্য, src0_তথ্য), nb1, nb01, row_আকার_bytes, 1)) {
            dma_queue_বের_করো(dma_q);
        }
    }
    dma_queue_flush(dma_q);
}



কাঠামো get_rows_flat_task {
    dma_addr_t src0_তথ্য;
    dma_addr_t dst_তথ্য;
};

স্থির অন্তঃরেখ কাঠামো get_rows_flat_task get_rows_flat_calc_task(
    ধ্রুবক কাঠামো htp_ops_context * octx,
    ধ্রুবক কাঠামো get_rows_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ধ্রুবক কাঠামো htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ধ্রুবক uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ধ্রুবক uint32_t rem = i - i12 * ne11 * ne10;
    ধ্রুবক uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ধ্রুবক uint32_t i10 = rem - i11 * ne10;

    কাঠামো get_rows_flat_task task;
    task.src0_তথ্য = 0;
    task.dst_তথ্য  = octx->dst->তথ্য + i10*nb1 + i11*nb2 + i12*nb3;

    যদি (need_src0) {
        ধ্রুবক uint32_t i01 = grctx->read_সূচক(grctx->src1_তথ্য, i, grctx->idx_type);
        প্রতিজ্ঞা(i01 < ne01);

        ধ্রুবক uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ধ্রুবক uint32_t i02 = i11 - q02 * ne02;
        ধ্রুবক uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ধ্রুবক uint32_t i03 = i12 - q03 * ne03;

        task.src0_তথ্য = octx->src[0]->তথ্য + i01*nb01 + i02*nb02 + i03*nb03;
    }

    ফেরত task;
}

স্থির শূন্য get_rows_thread_flat(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো get_rows_context * grctx = (কাঠামো get_rows_context *) তথ্য;
    কাঠামো htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    ধ্রুবক uint32_t dr  = grctx->tasks_per_thread;
    ধ্রুবক uint32_t ir0 = grctx->task_start + dr * ith;
    যদি (ir0 >= grctx->task_start + grctx->tasks) {
        ফেরত;
    }
    ধ্রুবক uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);

    dma_queue * dma_q = octx->ctx->dma[ith];
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    ধ্রুবক কাঠামো htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    ধ্রুবক uint32_t src0_bytes = grctx->src0_row_আকার;
    ধ্রুবক uint32_t dst_bytes  = grctx->dst_row_আকার;
    get_rows_flat_compute_fn_t compute = grctx->compute_flat;

    জন্য (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, spad_idx++) {
        ধ্রুবক uint32_t i = ir0 + step;
        কাঠামো get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, true);

        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(octx->dst->তথ্য,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_আকার),
                       dst_bytes, vtcm_layout->dst_spad_half_আকার, dst_bytes, 0);
        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_আকার, task.src0_তথ্য),
                       vtcm_layout->src0_spad_half_আকার, src0_bytes, src0_bytes, 1);
    }

    জন্য (uint32_t step = 0; step < ir1 - ir0; ++step) {
        ধ্রুবক uint32_t i = ir0 + step;
        শূন্য * dst_spad = (শূন্য *) dma_queue_বের_করো(dma_q).src;
        শূন্য * src_spad = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

        কাঠামো get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((দশমিক *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(task.dst_তথ্য, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_আকার, dst_bytes, 1);

        ধ্রুবক uint32_t পরবর্তী_step = step + 2;
        যদি (পরবর্তী_step < ir1 - ir0) {
            ধ্রুবক uint32_t pi = ir0 + পরবর্তী_step;
            কাঠামো get_rows_flat_task পরবর্তী_task = get_rows_flat_calc_task(octx, grctx, pi, true);
            dma_queue_ঠেলো(dma_q,
                           dma_make_তথ্য(src_spad, পরবর্তী_task.src0_তথ্য),
                           vtcm_layout->src0_spad_half_আকার, src0_bytes, src0_bytes, 1);
        }
    }

    dma_queue_flush(dma_q);
}



কাঠামো get_rows_tiled_task {
    dma_addr_t tile_src_base;
    dma_addr_t dst_তথ্য;
    uint32_t   row;
};

স্থির অন্তঃরেখ কাঠামো get_rows_tiled_task get_rows_tiled_calc_task(
    ধ্রুবক কাঠামো htp_ops_context * octx,
    ধ্রুবক কাঠামো get_rows_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_আকার
) {
    ধ্রুবক কাঠামো htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ধ্রুবক uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ধ্রুবক uint32_t rem = i - i12 * ne11 * ne10;
    ধ্রুবক uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ধ্রুবক uint32_t i10 = rem - i11 * ne10;
    ধ্রুবক uint32_t i01 = grctx->read_সূচক(grctx->src1_তথ্য, i, grctx->idx_type);
    প্রতিজ্ঞা(i01 < ne01);

    ধ্রুবক uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ধ্রুবক uint32_t i02 = i11 - q02 * ne02;
    ধ্রুবক uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ধ্রুবক uint32_t i03 = i12 - q03 * ne03;
    ধ্রুবক uint32_t column_tile = i01 / HTP_TENSOR_TILE_N_ROWS;
    ধ্রুবক uint32_t row = i01 % HTP_TENSOR_TILE_N_ROWS;
    ধ্রুবক dma_addr_t matrix = octx->src[0]->তথ্য + i02*nb02 + i03*nb03;

    কাঠামো get_rows_tiled_task task;
    task.tile_src_base = matrix + (column_tile * n_k_tiles) * tile_আকার;
    task.dst_তথ্য      = octx->dst->তথ্য + i10*nb1 + i11*nb2 + i12*nb3;
    task.row           = row;
    ফেরত task;
}

স্থির শূন্য get_rows_thread_tiled(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো get_rows_context * grctx = (কাঠামো get_rows_context *) তথ্য;
    কাঠামো htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    ধ্রুবক uint32_t dr  = grctx->tasks_per_thread;
    ধ্রুবক uint32_t ir0 = grctx->task_start + dr * ith;
    যদি (ir0 >= grctx->task_start + grctx->tasks) {
        ফেরত;
    }

    ধ্রুবক uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    ধ্রুবক uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ধ্রুবক কাঠামো htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * dst_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    dma_queue * dma_q = octx->ctx->dma[ith];
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুবক uint32_t tile_আকার   = grctx->tile_আকার;
    ধ্রুবক uint32_t tile_stride = grctx->tile_stride;
    ধ্রুবক uint32_t dst_bytes   = grctx->dst_row_আকার;
    get_rows_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    জন্য (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, ++spad_idx) {
        ধ্রুবক uint32_t i = ir0 + step;
        কাঠামো get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_আকার);

        // Dummy writeback to prime the queue with dst descriptor
        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(task.dst_তথ্য, dst_spad_base + spad_idx * vtcm_layout->dst_spad_half_আকার),
                       dst_bytes, vtcm_layout->dst_spad_half_আকার, dst_bytes, 0);

        // Prefetch row tiles
        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(src_spad_base + spad_idx * vtcm_layout->src0_spad_half_আকার, task.tile_src_base),
                       tile_stride, tile_আকার, tile_আকার, n_k_tiles);
    }

    জন্য (uint32_t step = 0; step < ir1 - ir0; ++step) {
        ধ্রুবক uint32_t i = ir0 + step;
        দশমিক * dst_spad   = (দশমিক *) dma_queue_বের_করো(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        কাঠামো get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_আকার);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.row);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(task.dst_তথ্য, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_আকার, dst_bytes, 1);

        ধ্রুবক uint32_t পরবর্তী_step = step + 2;
        যদি (পরবর্তী_step < ir1 - ir0) {
            ধ্রুবক uint32_t ni = ir0 + পরবর্তী_step;
            কাঠামো get_rows_tiled_task পরবর্তী_task = get_rows_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_আকার);
            dma_queue_ঠেলো(dma_q,
                           dma_make_তথ্য(src_spad, পরবর্তী_task.tile_src_base),
                           tile_stride, tile_আকার, tile_আকার, n_k_tiles);
        }
    }

    dma_queue_flush(dma_q);
}

স্থির __attribute__((noinline)) শূন্য get_rows_dma_src1(
    কাঠামো htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_আকার
) {
    ধ্রুবক uint32_t ne10 = octx->src[1]->ne[0];
    ধ্রুবক uint32_t ne11 = octx->src[1]->ne[1];
    ধ্রুবক uint32_t ne12 = octx->src[1]->ne[2];
    ধ্রুবক uint32_t ne13 = octx->src[1]->ne[3];
    ধ্রুবক uint32_t nb11 = octx->src[1]->nb[1];
    ধ্রুবক uint32_t nb12 = octx->src[1]->nb[2];
    ধ্রুবক uint32_t nb13 = octx->src[1]->nb[3];

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    জন্য (uint32_t i13 = 0; i13 < ne13; ++i13) {
        জন্য (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ধ্রুবক dma_addr_t src_plane = octx->src[1]->তথ্য + i12 * nb12 + i13 * nb13;
            যখন (!dma_queue_ঠেলো(dma_q, dma_make_তথ্য(cur_dst, src_plane),
                                   ne10 * elem_আকার, nb11, ne10 * elem_আকার, ne11)) {
                dma_queue_বের_করো(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_আকার;
        }
    }
    dma_queue_flush(dma_q);
}

পূর্ণসংখ্যা op_get_rows(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_get_rows_kernel_params * kparams = (ধ্রুবক কাঠামো htp_get_rows_kernel_params *) octx->kernel_params;

    ধ্রুবক uint32_t ne00          = octx->src[0]->ne[0];
    ধ্রুবক uint32_t src0_row_আকার = htp_tensor_get_row_আকার(octx->src[0]->type, ne00);
    যদি (src0_row_আকার == 0) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) {
        যদি (octx->src[0]->type != octx->dst->type) {
            ফেরত HTP_STATUS_NO_SUPPORT;
        }
    } অন্যথা {
        যদি (octx->dst->type != HTP_TYPE_F32) {
            ফেরত HTP_STATUS_NO_SUPPORT;
        }
    }

    যদি (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক কাঠামো htp_tensor * dst = octx->dst;
    ধ্রুবক uint32_t যোগ_tasks    = kparams->যোগ_tasks;
    ধ্রুবক uint32_t dst_row_আকার   = (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) ? src0_row_আকার : ne00 * আকার(দশমিক);

    uint32_t task_start = 0;
    uint32_t tasks      = যোগ_tasks;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t tasks_per_chunk = 1;
        htp_tensor_mdev_rows_per_chunk(dst, dst_row_আকার / dst->ne[0], dst_row_আকার, &tasks_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_tasks, tasks_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        task_start = range.start;
        tasks      = range.গুনতি;
    }

    যদি (tasks == 0) {
        ফেরত HTP_STATUS_OK;
    }

    যদি (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;

    কাঠামো get_rows_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;
    grctx.compute_flat = ফাঁকা;
    grctx.compute_tiled = ফাঁকা;
    grctx.task_start = task_start;
    grctx.tasks = tasks;
    grctx.tasks_per_thread = octx->ctx->mdev.গুনতি == 1 ? kparams->tasks_per_thread : fastdiv(tasks + n_threads - 1, &octx->n_threads_div);
    grctx.tile_আকার   = 0;
    grctx.tile_stride = 0;
    grctx.src0_row_আকার = src0_row_আকার;
    grctx.dst_row_আকার  = dst_row_আকার;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_rows_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, যোগ_tasks, grctx.idx_type, n_threads);

    যদি (grctx.vtcm_layout.যোগ_bytes > octx->ctx->vtcm_আকার) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_আকার, grctx.vtcm_layout.যোগ_bytes);
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    work_queue_func_t q_func = ফাঁকা;
    নির্বাচন (kparams->kernel_type) {
        ক্ষেত্র HTP_GET_ROWS_KERNEL_SAMETYPE:
            q_func = get_rows_thread_sametype;
            বিরতি;
        ক্ষেত্র HTP_GET_ROWS_KERNEL_TILED:
            নির্বাচন (octx->src[0]->type) {
                ক্ষেত্র HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_row_tiled_f16_f32;  বিরতি;
                ক্ষেত্র HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q1_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q2_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q8_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_k_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q5_k_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q6_k_f32; বিরতি;
                পূর্বনির্ধারিত:            ফেরত HTP_STATUS_NO_SUPPORT;
            }
            grctx.tile_আকার   = htp_tensor_get_tile_আকার(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_আকার(octx->src[0]->type);
            q_func = get_rows_thread_tiled;
            বিরতি;
        ক্ষেত্র HTP_GET_ROWS_KERNEL_FLAT:
            নির্বাচন (octx->src[0]->type) {
                ক্ষেত্র HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_row_flat_f16_f32;  বিরতি;
                ক্ষেত্র HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_row_flat_bf16_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_row_flat_q1_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_row_flat_q2_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_row_flat_q8_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_row_flat_q4_0_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_row_flat_q4_k_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_row_flat_q5_k_f32; বিরতি;
                ক্ষেত্র HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_row_flat_q6_k_f32; বিরতি;
                পূর্বনির্ধারিত:            ফেরত HTP_STATUS_NO_SUPPORT;
            }
            q_func = get_rows_thread_flat;
            বিরতি;
        পূর্বনির্ধারিত:
            ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t elem_আকার = (grctx.idx_type == HTP_TYPE_I64) ? আকার(int64_t) : আকার(int32_t);
    ধ্রুবক size_t src1_bytes  = (size_t) যোগ_tasks * elem_আকার;

    যদি (!htp_tensor_is_extended(octx->src[1]) && src1_bytes <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_আকার)) {
        grctx.src1_তথ্য  = (ধ্রুবক শূন্য *) (uintptr_t) octx->src[1]->তথ্য;
        grctx.read_সূচক = get_rows_read_সূচক_ddr;
    } অন্যথা {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_base, grctx.vtcm_layout.off_src1);
        get_rows_dma_src1(octx, vtcm_src1, elem_আকার);
        grctx.src1_তথ্য  = vtcm_src1;
        grctx.read_সূচক = get_rows_read_সূচক_vtcm;
    }

    FARF(উপরের_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         grctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         kparams->kernel_type, n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &grctx, n_threads);
    ফেরত HTP_STATUS_OK;
}
