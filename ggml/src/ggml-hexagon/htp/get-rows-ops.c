#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hvx-quant.h"
#अन्तर्भाव "get-rows-ops.h"
#अन्तर्भाव "work-queue.h"

संरचना get_rows_context;

प्रकारनिर्देश uint32_t (*get_rows_read_index_fn_t)(ध्रुव शून्य * src1, uint32_t idx, पूर्णाङ्क idx_type);
प्रकारनिर्देश शून्य (*get_rows_flat_compute_fn_t)(भिन्न * dst_spad, ध्रुव शून्य * src_spad, uint32_t cur_elems);
प्रकारनिर्देश शून्य (*get_rows_tiled_compute_fn_t)(भिन्न * dst_spad, ध्रुव uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row);

संरचना get_rows_context {
    संरचना htp_ops_context * octx;
    ध्रुव संरचना htp_get_rows_kernel_params * kparams;
    संरचना htp_get_rows_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_base;
    ध्रुव शून्य * src1_data;
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
    पूर्णाङ्क idx_type;
};

#define get_rows_preamble                      \
    ध्रुव uint32_t ne00 = octx->src[0]->ne[0]; \
    ध्रुव uint32_t ne01 = octx->src[0]->ne[1]; \
    ध्रुव uint32_t ne02 = octx->src[0]->ne[2]; \
    ध्रुव uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ध्रुव uint32_t ne10 = octx->src[1]->ne[0]; \
    ध्रुव uint32_t ne11 = octx->src[1]->ne[1]; \
    ध्रुव uint32_t ne12 = octx->src[1]->ne[2]; \
    ध्रुव uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ध्रुव uint32_t ne0 = octx->dst->ne[0];     \
    ध्रुव uint32_t ne1 = octx->dst->ne[1];     \
    ध्रुव uint32_t ne2 = octx->dst->ne[2];     \
    ध्रुव uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ध्रुव uint32_t nb01 = octx->src[0]->nb[1]; \
    ध्रुव uint32_t nb02 = octx->src[0]->nb[2]; \
    ध्रुव uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ध्रुव uint32_t nb10 = octx->src[1]->nb[0]; \
    ध्रुव uint32_t nb11 = octx->src[1]->nb[1]; \
    ध्रुव uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ध्रुव uint32_t nb1 = octx->dst->nb[1];     \
    ध्रुव uint32_t nb2 = octx->dst->nb[2];     \
    ध्रुव uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ध्रुव uint32_t nr = ne10 * ne11 * ne12;

स्थिर uint32_t get_rows_read_index_ddr(
    ध्रुव शून्य * src1,
    uint32_t idx,
    पूर्णाङ्क idx_type
) {
    ध्रुव uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? परिमाण(int64_t) : परिमाण(int32_t));
    प्रत्यागम *(ध्रुव uint32_t *) ((ध्रुव uint8_t *) src1 + off);
}

स्थिर __attribute__((noinline)) uint32_t get_rows_read_index_vtcm(
    ध्रुव शून्य * src1,
    uint32_t idx,
    पूर्णाङ्क idx_type
) {
    ध्रुव uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? परिमाण(int64_t) : परिमाण(int32_t));
    ध्रुव HVX_Vector v = hvx_vmem((ध्रुव uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    प्रत्यागम (uint32_t) Q6_R_vextract_VR(v, off);
}

स्थिर शून्य get_rows_thread_sametype(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना get_rows_context * grctx = (संरचना get_rows_context *) data;
    संरचना htp_ops_context * octx = grctx->octx;
    ध्रुव संरचना htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ध्रुव uint32_t dr  = grctx->tasks_per_thread;
    ध्रुव uint32_t ir0 = grctx->task_start + dr * ith;
    यदि (ir0 >= grctx->task_start + grctx->tasks) {
        प्रत्यागम;
    }
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    ध्रुव uint32_t row_size_bytes = grctx->src0_row_size;
    dma_queue * dma_q = octx->ctx->dma[ith];

    प्रति (uint32_t i = ir0; i < ir1; ++i) {
        ध्रुव uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ध्रुव uint32_t rem = i - i12 * ne11 * ne10;
        ध्रुव uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ध्रुव uint32_t i10 = rem - i11 * ne10;

        ध्रुव uint32_t i01 = grctx->read_index(grctx->src1_data, i, grctx->idx_type);
        प्रतिज्ञा(i01 < ne01);

        ध्रुव uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ध्रुव uint32_t i02 = i11 - q02 * ne02;
        ध्रुव uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ध्रुव uint32_t i03 = i12 - q03 * ne03;

        ध्रुव dma_addr_t src0_data = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;
        ध्रुव dma_addr_t dst_data  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3;

        यावत् (!dma_queue_धकेलो(dma_q, dma_make_data(dst_data, src0_data), nb1, nb01, row_size_bytes, 1)) {
            dma_queue_निकालो(dma_q);
        }
    }
    dma_queue_flush(dma_q);
}



संरचना get_rows_flat_task {
    dma_addr_t src0_data;
    dma_addr_t dst_data;
};

स्थिर अन्तःस्थ संरचना get_rows_flat_task get_rows_flat_calc_task(
    ध्रुव संरचना htp_ops_context * octx,
    ध्रुव संरचना get_rows_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ध्रुव संरचना htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ध्रुव uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ध्रुव uint32_t rem = i - i12 * ne11 * ne10;
    ध्रुव uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ध्रुव uint32_t i10 = rem - i11 * ne10;

    संरचना get_rows_flat_task task;
    task.src0_data = 0;
    task.dst_data  = octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3;

    यदि (need_src0) {
        ध्रुव uint32_t i01 = grctx->read_index(grctx->src1_data, i, grctx->idx_type);
        प्रतिज्ञा(i01 < ne01);

        ध्रुव uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ध्रुव uint32_t i02 = i11 - q02 * ne02;
        ध्रुव uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ध्रुव uint32_t i03 = i12 - q03 * ne03;

        task.src0_data = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;
    }

    प्रत्यागम task;
}

स्थिर शून्य get_rows_thread_flat(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना get_rows_context * grctx = (संरचना get_rows_context *) data;
    संरचना htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    ध्रुव uint32_t dr  = grctx->tasks_per_thread;
    ध्रुव uint32_t ir0 = grctx->task_start + dr * ith;
    यदि (ir0 >= grctx->task_start + grctx->tasks) {
        प्रत्यागम;
    }
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);

    dma_queue * dma_q = octx->ctx->dma[ith];
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    ध्रुव संरचना htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    ध्रुव uint32_t src0_bytes = grctx->src0_row_size;
    ध्रुव uint32_t dst_bytes  = grctx->dst_row_size;
    get_rows_flat_compute_fn_t compute = grctx->compute_flat;

    प्रति (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, spad_idx++) {
        ध्रुव uint32_t i = ir0 + step;
        संरचना get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, true);

        dma_queue_धकेलो(dma_q,
                       dma_make_data(octx->dst->data,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 0);
        dma_queue_धकेलो(dma_q,
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size, task.src0_data),
                       vtcm_layout->src0_spad_half_size, src0_bytes, src0_bytes, 1);
    }

    प्रति (uint32_t step = 0; step < ir1 - ir0; ++step) {
        ध्रुव uint32_t i = ir0 + step;
        शून्य * dst_spad = (शून्य *) dma_queue_निकालो(dma_q).src;
        शून्य * src_spad = (शून्य *) dma_queue_निकालो(dma_q).dst;

        संरचना get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((भिन्न *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_queue_धकेलो(dma_q,
                       dma_make_data(task.dst_data, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 1);

        ध्रुव uint32_t next_step = step + 2;
        यदि (next_step < ir1 - ir0) {
            ध्रुव uint32_t pi = ir0 + next_step;
            संरचना get_rows_flat_task next_task = get_rows_flat_calc_task(octx, grctx, pi, true);
            dma_queue_धकेलो(dma_q,
                           dma_make_data(src_spad, next_task.src0_data),
                           vtcm_layout->src0_spad_half_size, src0_bytes, src0_bytes, 1);
        }
    }

    dma_queue_flush(dma_q);
}



संरचना get_rows_tiled_task {
    dma_addr_t tile_src_base;
    dma_addr_t dst_data;
    uint32_t   row;
};

स्थिर अन्तःस्थ संरचना get_rows_tiled_task get_rows_tiled_calc_task(
    ध्रुव संरचना htp_ops_context * octx,
    ध्रुव संरचना get_rows_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_size
) {
    ध्रुव संरचना htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    ध्रुव uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ध्रुव uint32_t rem = i - i12 * ne11 * ne10;
    ध्रुव uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ध्रुव uint32_t i10 = rem - i11 * ne10;
    ध्रुव uint32_t i01 = grctx->read_index(grctx->src1_data, i, grctx->idx_type);
    प्रतिज्ञा(i01 < ne01);

    ध्रुव uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ध्रुव uint32_t i02 = i11 - q02 * ne02;
    ध्रुव uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ध्रुव uint32_t i03 = i12 - q03 * ne03;
    ध्रुव uint32_t column_tile = i01 / HTP_TENSOR_TILE_N_ROWS;
    ध्रुव uint32_t row = i01 % HTP_TENSOR_TILE_N_ROWS;
    ध्रुव dma_addr_t matrix = octx->src[0]->data + i02*nb02 + i03*nb03;

    संरचना get_rows_tiled_task task;
    task.tile_src_base = matrix + (column_tile * n_k_tiles) * tile_size;
    task.dst_data      = octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3;
    task.row           = row;
    प्रत्यागम task;
}

स्थिर शून्य get_rows_thread_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना get_rows_context * grctx = (संरचना get_rows_context *) data;
    संरचना htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    ध्रुव uint32_t dr  = grctx->tasks_per_thread;
    ध्रुव uint32_t ir0 = grctx->task_start + dr * ith;
    यदि (ir0 >= grctx->task_start + grctx->tasks) {
        प्रत्यागम;
    }

    ध्रुव uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    ध्रुव uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ध्रुव संरचना htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * dst_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    dma_queue * dma_q = octx->ctx->dma[ith];
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t tile_size   = grctx->tile_size;
    ध्रुव uint32_t tile_stride = grctx->tile_stride;
    ध्रुव uint32_t dst_bytes   = grctx->dst_row_size;
    get_rows_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    प्रति (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, ++spad_idx) {
        ध्रुव uint32_t i = ir0 + step;
        संरचना get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_size);

        // Dummy writeback to prime the queue with dst descriptor
        dma_queue_धकेलो(dma_q,
                       dma_make_data(task.dst_data, dst_spad_base + spad_idx * vtcm_layout->dst_spad_half_size),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 0);

        // Prefetch row tiles
        dma_queue_धकेलो(dma_q,
                       dma_make_data(src_spad_base + spad_idx * vtcm_layout->src0_spad_half_size, task.tile_src_base),
                       tile_stride, tile_size, tile_size, n_k_tiles);
    }

    प्रति (uint32_t step = 0; step < ir1 - ir0; ++step) {
        ध्रुव uint32_t i = ir0 + step;
        भिन्न * dst_spad   = (भिन्न *) dma_queue_निकालो(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        संरचना get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_size);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.row);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_queue_धकेलो(dma_q,
                       dma_make_data(task.dst_data, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_size, dst_bytes, 1);

        ध्रुव uint32_t next_step = step + 2;
        यदि (next_step < ir1 - ir0) {
            ध्रुव uint32_t ni = ir0 + next_step;
            संरचना get_rows_tiled_task next_task = get_rows_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_size);
            dma_queue_धकेलो(dma_q,
                           dma_make_data(src_spad, next_task.tile_src_base),
                           tile_stride, tile_size, tile_size, n_k_tiles);
        }
    }

    dma_queue_flush(dma_q);
}

स्थिर __attribute__((noinline)) शून्य get_rows_dma_src1(
    संरचना htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_size
) {
    ध्रुव uint32_t ne10 = octx->src[1]->ne[0];
    ध्रुव uint32_t ne11 = octx->src[1]->ne[1];
    ध्रुव uint32_t ne12 = octx->src[1]->ne[2];
    ध्रुव uint32_t ne13 = octx->src[1]->ne[3];
    ध्रुव uint32_t nb11 = octx->src[1]->nb[1];
    ध्रुव uint32_t nb12 = octx->src[1]->nb[2];
    ध्रुव uint32_t nb13 = octx->src[1]->nb[3];

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    प्रति (uint32_t i13 = 0; i13 < ne13; ++i13) {
        प्रति (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ध्रुव dma_addr_t src_plane = octx->src[1]->data + i12 * nb12 + i13 * nb13;
            यावत् (!dma_queue_धकेलो(dma_q, dma_make_data(cur_dst, src_plane),
                                   ne10 * elem_size, nb11, ne10 * elem_size, ne11)) {
                dma_queue_निकालो(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_size;
        }
    }
    dma_queue_flush(dma_q);
}

पूर्णाङ्क op_get_rows(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_get_rows_kernel_params * kparams = (ध्रुव संरचना htp_get_rows_kernel_params *) octx->kernel_params;

    ध्रुव uint32_t ne00          = octx->src[0]->ne[0];
    ध्रुव uint32_t src0_row_size = htp_tensor_get_row_size(octx->src[0]->type, ne00);
    यदि (src0_row_size == 0) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    यदि (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) {
        यदि (octx->src[0]->type != octx->dst->type) {
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
        }
    } अन्यथा {
        यदि (octx->dst->type != HTP_TYPE_F32) {
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
        }
    }

    यदि (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव संरचना htp_tensor * dst = octx->dst;
    ध्रुव uint32_t total_tasks    = kparams->total_tasks;
    ध्रुव uint32_t dst_row_size   = (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) ? src0_row_size : ne00 * परिमाण(भिन्न);

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    यदि (octx->ctx->mdev.गिनती > 1) {
        uint32_t tasks_per_chunk = 1;
        htp_tensor_mdev_rows_per_chunk(dst, dst_row_size / dst->ne[0], dst_row_size, &tasks_per_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, tasks_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        task_start = range.start;
        tasks      = range.गिनती;
    }

    यदि (tasks == 0) {
        प्रत्यागम HTP_STATUS_OK;
    }

    यदि (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        प्रत्यागम HTP_STATUS_INVAL_PARAMS;
    }

    ध्रुव uint32_t n_threads = octx->n_threads;

    संरचना get_rows_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;
    grctx.compute_flat = रिक्त;
    grctx.compute_tiled = रिक्त;
    grctx.task_start = task_start;
    grctx.tasks = tasks;
    grctx.tasks_per_thread = octx->ctx->mdev.गिनती == 1 ? kparams->tasks_per_thread : fastdiv(tasks + n_threads - 1, &octx->n_threads_div);
    grctx.tile_size   = 0;
    grctx.tile_stride = 0;
    grctx.src0_row_size = src0_row_size;
    grctx.dst_row_size  = dst_row_size;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_rows_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, total_tasks, grctx.idx_type, n_threads);

    यदि (grctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, grctx.vtcm_layout.total_bytes);
        प्रत्यागम HTP_STATUS_INVAL_PARAMS;
    }

    work_queue_func_t q_func = रिक्त;
    विकल्प (kparams->kernel_type) {
        पक्ष HTP_GET_ROWS_KERNEL_SAMETYPE:
            q_func = get_rows_thread_sametype;
            भङ्ग;
        पक्ष HTP_GET_ROWS_KERNEL_TILED:
            विकल्प (octx->src[0]->type) {
                पक्ष HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_row_tiled_f16_f32;  भङ्ग;
                पक्ष HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q1_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q2_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q8_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_k_f32; भङ्ग;
                पक्ष HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q5_k_f32; भङ्ग;
                पक्ष HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q6_k_f32; भङ्ग;
                स्वभाव:            प्रत्यागम HTP_STATUS_NO_SUPPORT;
            }
            grctx.tile_size   = htp_tensor_get_tile_size(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_size(octx->src[0]->type);
            q_func = get_rows_thread_tiled;
            भङ्ग;
        पक्ष HTP_GET_ROWS_KERNEL_FLAT:
            विकल्प (octx->src[0]->type) {
                पक्ष HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_row_flat_f16_f32;  भङ्ग;
                पक्ष HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_row_flat_bf16_f32; भङ्ग;
                पक्ष HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_row_flat_q1_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_row_flat_q2_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_row_flat_q8_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_row_flat_q4_0_f32; भङ्ग;
                पक्ष HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_row_flat_q4_k_f32; भङ्ग;
                पक्ष HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_row_flat_q5_k_f32; भङ्ग;
                पक्ष HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_row_flat_q6_k_f32; भङ्ग;
                स्वभाव:            प्रत्यागम HTP_STATUS_NO_SUPPORT;
            }
            q_func = get_rows_thread_flat;
            भङ्ग;
        स्वभाव:
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव uint32_t elem_size = (grctx.idx_type == HTP_TYPE_I64) ? परिमाण(int64_t) : परिमाण(int32_t);
    ध्रुव माप_प्रकार src1_bytes  = (माप_प्रकार) total_tasks * elem_size;

    यदि (!htp_tensor_is_extended(octx->src[1]) && src1_bytes <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_size)) {
        grctx.src1_data  = (ध्रुव शून्य *) (uintptr_t) octx->src[1]->data;
        grctx.read_index = get_rows_read_index_ddr;
    } अन्यथा {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_base, grctx.vtcm_layout.off_src1);
        get_rows_dma_src1(octx, vtcm_src1, elem_size);
        grctx.src1_data  = vtcm_src1;
        grctx.read_index = get_rows_read_index_vtcm;
    }

    FARF(ऊपरी_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         grctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         kparams->kernel_type, n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &grctx, n_threads);
    प्रत्यागम HTP_STATUS_OK;
}
