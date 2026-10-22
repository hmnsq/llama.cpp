#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#समावेश <math.h>
#समावेश <string.h>

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "hex-common.h"
#समावेश "dma-queue.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "hvx-utils.h"
#समावेश "hvx-quant.h"
#समावेश "get-rows-ops.h"
#समावेश "work-queue.h"

काष्ठा get_rows_context;

प्रकार uint32_t (*get_rows_read_सूचकांक_fn_t)(स्थिर व्योम * src1, uint32_t idx, पूर्णांक idx_type);
प्रकार व्योम (*get_rows_flat_compute_fn_t)(भग्न * dst_spad, स्थिर व्योम * src_spad, uint32_t cur_elems);
प्रकार व्योम (*get_rows_tiled_compute_fn_t)(भग्न * dst_spad, स्थिर uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row);

काष्ठा get_rows_context {
    काष्ठा htp_ops_context * octx;
    स्थिर काष्ठा htp_get_rows_kernel_params * kparams;
    काष्ठा htp_get_rows_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_base;
    स्थिर व्योम * src1_तथ्य;
    get_rows_read_सूचकांक_fn_t    read_सूचकांक;
    get_rows_flat_compute_fn_t  compute_flat;
    get_rows_tiled_compute_fn_t compute_tiled;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
    uint32_t tile_आकार;
    uint32_t tile_stride;
    uint32_t src0_row_आकार;
    uint32_t dst_row_आकार;
    पूर्णांक idx_type;
};

#घोषणा get_rows_preamble                      \
    स्थिर uint32_t ne00 = octx->src[0]->ne[0]; \
    स्थिर uint32_t ne01 = octx->src[0]->ne[1]; \
    स्थिर uint32_t ne02 = octx->src[0]->ne[2]; \
    स्थिर uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    स्थिर uint32_t ne10 = octx->src[1]->ne[0]; \
    स्थिर uint32_t ne11 = octx->src[1]->ne[1]; \
    स्थिर uint32_t ne12 = octx->src[1]->ne[2]; \
    स्थिर uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    स्थिर uint32_t ne0 = octx->dst->ne[0];     \
    स्थिर uint32_t ne1 = octx->dst->ne[1];     \
    स्थिर uint32_t ne2 = octx->dst->ne[2];     \
    स्थिर uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    स्थिर uint32_t nb01 = octx->src[0]->nb[1]; \
    स्थिर uint32_t nb02 = octx->src[0]->nb[2]; \
    स्थिर uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    स्थिर uint32_t nb10 = octx->src[1]->nb[0]; \
    स्थिर uint32_t nb11 = octx->src[1]->nb[1]; \
    स्थिर uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    स्थिर uint32_t nb1 = octx->dst->nb[1];     \
    स्थिर uint32_t nb2 = octx->dst->nb[2];     \
    स्थिर uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    स्थिर uint32_t nr = ne10 * ne11 * ne12;

अटल uint32_t get_rows_read_सूचकांक_ddr(
    स्थिर व्योम * src1,
    uint32_t idx,
    पूर्णांक idx_type
) {
    स्थिर uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? माप(int64_t) : माप(int32_t));
    वापस *(स्थिर uint32_t *) ((स्थिर uint8_t *) src1 + off);
}

अटल __attribute__((noinline)) uint32_t get_rows_read_सूचकांक_vtcm(
    स्थिर व्योम * src1,
    uint32_t idx,
    पूर्णांक idx_type
) {
    स्थिर uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? माप(int64_t) : माप(int32_t));
    स्थिर HVX_Vector v = hvx_vmem((स्थिर uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    वापस (uint32_t) Q6_R_vextract_VR(v, off);
}

अटल व्योम get_rows_thread_sametype(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा get_rows_context * grctx = (काष्ठा get_rows_context *) तथ्य;
    काष्ठा htp_ops_context * octx = grctx->octx;
    स्थिर काष्ठा htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    स्थिर uint32_t dr  = grctx->tasks_per_thread;
    स्थिर uint32_t ir0 = grctx->task_start + dr * ith;
    अगर (ir0 >= grctx->task_start + grctx->tasks) {
        वापस;
    }
    स्थिर uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    स्थिर uint32_t row_आकार_bytes = grctx->src0_row_आकार;
    dma_queue * dma_q = octx->ctx->dma[ith];

    क्रम (uint32_t i = ir0; i < ir1; ++i) {
        स्थिर uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        स्थिर uint32_t rem = i - i12 * ne11 * ne10;
        स्थिर uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        स्थिर uint32_t i10 = rem - i11 * ne10;

        स्थिर uint32_t i01 = grctx->read_सूचकांक(grctx->src1_तथ्य, i, grctx->idx_type);
        निश्चित(i01 < ne01);

        स्थिर uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        स्थिर uint32_t i02 = i11 - q02 * ne02;
        स्थिर uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        स्थिर uint32_t i03 = i12 - q03 * ne03;

        स्थिर dma_addr_t src0_तथ्य = octx->src[0]->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;
        स्थिर dma_addr_t dst_तथ्य  = octx->dst->तथ्य    + i10*nb1  + i11*nb2  + i12*nb3;

        जबतक (!dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_तथ्य, src0_तथ्य), nb1, nb01, row_आकार_bytes, 1)) {
            dma_queue_निकालो(dma_q);
        }
    }
    dma_queue_flush(dma_q);
}



काष्ठा get_rows_flat_task {
    dma_addr_t src0_तथ्य;
    dma_addr_t dst_तथ्य;
};

अटल अंतरभूत काष्ठा get_rows_flat_task get_rows_flat_calc_task(
    स्थिर काष्ठा htp_ops_context * octx,
    स्थिर काष्ठा get_rows_context * grctx,
    uint32_t i,
    bool need_src0
) {
    स्थिर काष्ठा htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    स्थिर uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    स्थिर uint32_t rem = i - i12 * ne11 * ne10;
    स्थिर uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    स्थिर uint32_t i10 = rem - i11 * ne10;

    काष्ठा get_rows_flat_task task;
    task.src0_तथ्य = 0;
    task.dst_तथ्य  = octx->dst->तथ्य + i10*nb1 + i11*nb2 + i12*nb3;

    अगर (need_src0) {
        स्थिर uint32_t i01 = grctx->read_सूचकांक(grctx->src1_तथ्य, i, grctx->idx_type);
        निश्चित(i01 < ne01);

        स्थिर uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        स्थिर uint32_t i02 = i11 - q02 * ne02;
        स्थिर uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        स्थिर uint32_t i03 = i12 - q03 * ne03;

        task.src0_तथ्य = octx->src[0]->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;
    }

    वापस task;
}

अटल व्योम get_rows_thread_flat(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा get_rows_context * grctx = (काष्ठा get_rows_context *) तथ्य;
    काष्ठा htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    स्थिर uint32_t dr  = grctx->tasks_per_thread;
    स्थिर uint32_t ir0 = grctx->task_start + dr * ith;
    अगर (ir0 >= grctx->task_start + grctx->tasks) {
        वापस;
    }
    स्थिर uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);

    dma_queue * dma_q = octx->ctx->dma[ith];
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    स्थिर काष्ठा htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    स्थिर uint32_t src0_bytes = grctx->src0_row_आकार;
    स्थिर uint32_t dst_bytes  = grctx->dst_row_आकार;
    get_rows_flat_compute_fn_t compute = grctx->compute_flat;

    क्रम (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, spad_idx++) {
        स्थिर uint32_t i = ir0 + step;
        काष्ठा get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, true);

        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(octx->dst->तथ्य,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_आकार),
                       dst_bytes, vtcm_layout->dst_spad_half_आकार, dst_bytes, 0);
        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_आकार, task.src0_तथ्य),
                       vtcm_layout->src0_spad_half_आकार, src0_bytes, src0_bytes, 1);
    }

    क्रम (uint32_t step = 0; step < ir1 - ir0; ++step) {
        स्थिर uint32_t i = ir0 + step;
        व्योम * dst_spad = (व्योम *) dma_queue_निकालो(dma_q).src;
        व्योम * src_spad = (व्योम *) dma_queue_निकालो(dma_q).dst;

        काष्ठा get_rows_flat_task task = get_rows_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((भग्न *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(task.dst_तथ्य, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_आकार, dst_bytes, 1);

        स्थिर uint32_t अग्र_step = step + 2;
        अगर (अग्र_step < ir1 - ir0) {
            स्थिर uint32_t pi = ir0 + अग्र_step;
            काष्ठा get_rows_flat_task अग्र_task = get_rows_flat_calc_task(octx, grctx, pi, true);
            dma_queue_धकेलो(dma_q,
                           dma_make_तथ्य(src_spad, अग्र_task.src0_तथ्य),
                           vtcm_layout->src0_spad_half_आकार, src0_bytes, src0_bytes, 1);
        }
    }

    dma_queue_flush(dma_q);
}



काष्ठा get_rows_tiled_task {
    dma_addr_t tile_src_base;
    dma_addr_t dst_तथ्य;
    uint32_t   row;
};

अटल अंतरभूत काष्ठा get_rows_tiled_task get_rows_tiled_calc_task(
    स्थिर काष्ठा htp_ops_context * octx,
    स्थिर काष्ठा get_rows_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_आकार
) {
    स्थिर काष्ठा htp_get_rows_kernel_params * kparams = grctx->kparams;
    get_rows_preamble;

    स्थिर uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    स्थिर uint32_t rem = i - i12 * ne11 * ne10;
    स्थिर uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    स्थिर uint32_t i10 = rem - i11 * ne10;
    स्थिर uint32_t i01 = grctx->read_सूचकांक(grctx->src1_तथ्य, i, grctx->idx_type);
    निश्चित(i01 < ne01);

    स्थिर uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    स्थिर uint32_t i02 = i11 - q02 * ne02;
    स्थिर uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    स्थिर uint32_t i03 = i12 - q03 * ne03;
    स्थिर uint32_t column_tile = i01 / HTP_TENSOR_TILE_N_ROWS;
    स्थिर uint32_t row = i01 % HTP_TENSOR_TILE_N_ROWS;
    स्थिर dma_addr_t matrix = octx->src[0]->तथ्य + i02*nb02 + i03*nb03;

    काष्ठा get_rows_tiled_task task;
    task.tile_src_base = matrix + (column_tile * n_k_tiles) * tile_आकार;
    task.dst_तथ्य      = octx->dst->तथ्य + i10*nb1 + i11*nb2 + i12*nb3;
    task.row           = row;
    वापस task;
}

अटल व्योम get_rows_thread_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा get_rows_context * grctx = (काष्ठा get_rows_context *) तथ्य;
    काष्ठा htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    स्थिर uint32_t dr  = grctx->tasks_per_thread;
    स्थिर uint32_t ir0 = grctx->task_start + dr * ith;
    अगर (ir0 >= grctx->task_start + grctx->tasks) {
        वापस;
    }

    स्थिर uint32_t ir1 = MIN(ir0 + dr, grctx->task_start + grctx->tasks);
    स्थिर uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    स्थिर काष्ठा htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_thread;
    uint8_t * dst_spad_base = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_thread;
    dma_queue * dma_q = octx->ctx->dma[ith];
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर uint32_t tile_आकार   = grctx->tile_आकार;
    स्थिर uint32_t tile_stride = grctx->tile_stride;
    स्थिर uint32_t dst_bytes   = grctx->dst_row_आकार;
    get_rows_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    क्रम (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, ++spad_idx) {
        स्थिर uint32_t i = ir0 + step;
        काष्ठा get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_आकार);

        // Dummy writeback to prime the queue with dst descriptor
        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(task.dst_तथ्य, dst_spad_base + spad_idx * vtcm_layout->dst_spad_half_आकार),
                       dst_bytes, vtcm_layout->dst_spad_half_आकार, dst_bytes, 0);

        // Prefetch row tiles
        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(src_spad_base + spad_idx * vtcm_layout->src0_spad_half_आकार, task.tile_src_base),
                       tile_stride, tile_आकार, tile_आकार, n_k_tiles);
    }

    क्रम (uint32_t step = 0; step < ir1 - ir0; ++step) {
        स्थिर uint32_t i = ir0 + step;
        भग्न * dst_spad   = (भग्न *) dma_queue_निकालो(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        काष्ठा get_rows_tiled_task task = get_rows_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_आकार);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.row);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(task.dst_तथ्य, dst_spad),
                       dst_bytes, vtcm_layout->dst_spad_half_आकार, dst_bytes, 1);

        स्थिर uint32_t अग्र_step = step + 2;
        अगर (अग्र_step < ir1 - ir0) {
            स्थिर uint32_t ni = ir0 + अग्र_step;
            काष्ठा get_rows_tiled_task अग्र_task = get_rows_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_आकार);
            dma_queue_धकेलो(dma_q,
                           dma_make_तथ्य(src_spad, अग्र_task.tile_src_base),
                           tile_stride, tile_आकार, tile_आकार, n_k_tiles);
        }
    }

    dma_queue_flush(dma_q);
}

अटल __attribute__((noinline)) व्योम get_rows_dma_src1(
    काष्ठा htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_आकार
) {
    स्थिर uint32_t ne10 = octx->src[1]->ne[0];
    स्थिर uint32_t ne11 = octx->src[1]->ne[1];
    स्थिर uint32_t ne12 = octx->src[1]->ne[2];
    स्थिर uint32_t ne13 = octx->src[1]->ne[3];
    स्थिर uint32_t nb11 = octx->src[1]->nb[1];
    स्थिर uint32_t nb12 = octx->src[1]->nb[2];
    स्थिर uint32_t nb13 = octx->src[1]->nb[3];

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    क्रम (uint32_t i13 = 0; i13 < ne13; ++i13) {
        क्रम (uint32_t i12 = 0; i12 < ne12; ++i12) {
            स्थिर dma_addr_t src_plane = octx->src[1]->तथ्य + i12 * nb12 + i13 * nb13;
            जबतक (!dma_queue_धकेलो(dma_q, dma_make_तथ्य(cur_dst, src_plane),
                                   ne10 * elem_आकार, nb11, ne10 * elem_आकार, ne11)) {
                dma_queue_निकालो(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_आकार;
        }
    }
    dma_queue_flush(dma_q);
}

पूर्णांक op_get_rows(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_get_rows_kernel_params * kparams = (स्थिर काष्ठा htp_get_rows_kernel_params *) octx->kernel_params;

    स्थिर uint32_t ne00          = octx->src[0]->ne[0];
    स्थिर uint32_t src0_row_आकार = htp_tensor_get_row_आकार(octx->src[0]->type, ne00);
    अगर (src0_row_आकार == 0) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) {
        अगर (octx->src[0]->type != octx->dst->type) {
            वापस HTP_STATUS_NO_SUPPORT;
        }
    } अन्यथा {
        अगर (octx->dst->type != HTP_TYPE_F32) {
            वापस HTP_STATUS_NO_SUPPORT;
        }
    }

    अगर (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    स्थिर uint32_t योग_tasks    = kparams->योग_tasks;
    स्थिर uint32_t dst_row_आकार   = (kparams->kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) ? src0_row_आकार : ne00 * माप(भग्न);

    uint32_t task_start = 0;
    uint32_t tasks      = योग_tasks;

    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t tasks_per_chunk = 1;
        htp_tensor_mdev_rows_per_chunk(dst, dst_row_आकार / dst->ne[0], dst_row_आकार, &tasks_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_tasks, tasks_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        task_start = range.start;
        tasks      = range.गिनती;
    }

    अगर (tasks == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर uint32_t n_threads = octx->n_threads;

    काष्ठा get_rows_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;
    grctx.compute_flat = शून्य;
    grctx.compute_tiled = शून्य;
    grctx.task_start = task_start;
    grctx.tasks = tasks;
    grctx.tasks_per_thread = octx->ctx->mdev.गिनती == 1 ? kparams->tasks_per_thread : fastdiv(tasks + n_threads - 1, &octx->n_threads_div);
    grctx.tile_आकार   = 0;
    grctx.tile_stride = 0;
    grctx.src0_row_आकार = src0_row_आकार;
    grctx.dst_row_आकार  = dst_row_आकार;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_rows_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, योग_tasks, grctx.idx_type, n_threads);

    अगर (grctx.vtcm_layout.योग_bytes > octx->ctx->vtcm_आकार) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_आकार, grctx.vtcm_layout.योग_bytes);
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    work_queue_func_t q_func = शून्य;
    चयन (kparams->kernel_type) {
        हाल HTP_GET_ROWS_KERNEL_SAMETYPE:
            q_func = get_rows_thread_sametype;
            अवरोध;
        हाल HTP_GET_ROWS_KERNEL_TILED:
            चयन (octx->src[0]->type) {
                हाल HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_row_tiled_f16_f32;  अवरोध;
                हाल HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q1_0_f32; अवरोध;
                हाल HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q2_0_f32; अवरोध;
                हाल HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q8_0_f32; अवरोध;
                हाल HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_0_f32; अवरोध;
                हाल HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q4_k_f32; अवरोध;
                हाल HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q5_k_f32; अवरोध;
                हाल HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_row_tiled_q6_k_f32; अवरोध;
                शेष:            वापस HTP_STATUS_NO_SUPPORT;
            }
            grctx.tile_आकार   = htp_tensor_get_tile_आकार(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_आकार(octx->src[0]->type);
            q_func = get_rows_thread_tiled;
            अवरोध;
        हाल HTP_GET_ROWS_KERNEL_FLAT:
            चयन (octx->src[0]->type) {
                हाल HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_row_flat_f16_f32;  अवरोध;
                हाल HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_row_flat_bf16_f32; अवरोध;
                हाल HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_row_flat_q1_0_f32; अवरोध;
                हाल HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_row_flat_q2_0_f32; अवरोध;
                हाल HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_row_flat_q8_0_f32; अवरोध;
                हाल HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_row_flat_q4_0_f32; अवरोध;
                हाल HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_row_flat_q4_k_f32; अवरोध;
                हाल HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_row_flat_q5_k_f32; अवरोध;
                हाल HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_row_flat_q6_k_f32; अवरोध;
                शेष:            वापस HTP_STATUS_NO_SUPPORT;
            }
            q_func = get_rows_thread_flat;
            अवरोध;
        शेष:
            वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर uint32_t elem_आकार = (grctx.idx_type == HTP_TYPE_I64) ? माप(int64_t) : माप(int32_t);
    स्थिर माप_प्रकार src1_bytes  = (माप_प्रकार) योग_tasks * elem_आकार;

    अगर (!htp_tensor_is_extended(octx->src[1]) && src1_bytes <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_आकार)) {
        grctx.src1_तथ्य  = (स्थिर व्योम *) (uintptr_t) octx->src[1]->तथ्य;
        grctx.read_सूचकांक = get_rows_read_सूचकांक_ddr;
    } अन्यथा {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_base, grctx.vtcm_layout.off_src1);
        get_rows_dma_src1(octx, vtcm_src1, elem_आकार);
        grctx.src1_तथ्य  = vtcm_src1;
        grctx.read_सूचकांक = get_rows_read_सूचकांक_vtcm;
    }

    FARF(ऊपरी_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         grctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         kparams->kernel_type, n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &grctx, n_threads);
    वापस HTP_STATUS_OK;
}
