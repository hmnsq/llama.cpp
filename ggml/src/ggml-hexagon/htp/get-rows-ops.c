#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <math.h>
#شامل <string.h>

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "dma-queue.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-utils.h"
#شامل "hvx-quant.h"
#شامل "get-rows-ops.h"
#شامل "work-queue.h"

ساخت get_صفیں_context;

قسمتعریف uint32_t (*get_صفیں_read_اشاریہ_fn_t)(ثابت خالی * src1, uint32_t idx, صحیح idx_type);
قسمتعریف خالی (*get_صفیں_flat_compute_fn_t)(اعشاری * dst_spad, ثابت خالی * src_spad, uint32_t cur_elems);
قسمتعریف خالی (*get_صفیں_tiled_compute_fn_t)(اعشاری * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف);

ساخت get_صفیں_context {
    ساخت htp_ops_context * octx;
    ثابت ساخت htp_get_صفیں_kernel_params * kparams;
    ساخت htp_get_صفیں_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_بنیاد;
    ثابت خالی * src1_ڈیٹا;
    get_صفیں_read_اشاریہ_fn_t    read_اشاریہ;
    get_صفیں_flat_compute_fn_t  compute_flat;
    get_صفیں_tiled_compute_fn_t compute_tiled;
    uint32_t task_آغاز;
    uint32_t tasks;
    uint32_t tasks_فی_thread;
    uint32_t tile_سائز;
    uint32_t tile_stride;
    uint32_t src0_صف_سائز;
    uint32_t dst_صف_سائز;
    صحیح idx_type;
};

#تعریف_کرو get_صفیں_preamble                      \
    ثابت uint32_t ne00 = octx->src[0]->ne[0]; \
    ثابت uint32_t ne01 = octx->src[0]->ne[1]; \
    ثابت uint32_t ne02 = octx->src[0]->ne[2]; \
    ثابت uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ثابت uint32_t ne10 = octx->src[1]->ne[0]; \
    ثابت uint32_t ne11 = octx->src[1]->ne[1]; \
    ثابت uint32_t ne12 = octx->src[1]->ne[2]; \
    ثابت uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ثابت uint32_t ne0 = octx->dst->ne[0];     \
    ثابت uint32_t ne1 = octx->dst->ne[1];     \
    ثابت uint32_t ne2 = octx->dst->ne[2];     \
    ثابت uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ثابت uint32_t nb01 = octx->src[0]->nb[1]; \
    ثابت uint32_t nb02 = octx->src[0]->nb[2]; \
    ثابت uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ثابت uint32_t nb10 = octx->src[1]->nb[0]; \
    ثابت uint32_t nb11 = octx->src[1]->nb[1]; \
    ثابت uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ثابت uint32_t nb1 = octx->dst->nb[1];     \
    ثابت uint32_t nb2 = octx->dst->nb[2];     \
    ثابت uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ثابت uint32_t nr = ne10 * ne11 * ne12;

ساکن uint32_t get_صفیں_read_اشاریہ_ddr(
    ثابت خالی * src1,
    uint32_t idx,
    صحیح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    واپس *(ثابت uint32_t *) ((ثابت uint8_t *) src1 + off);
}

ساکن __attribute__((noinline)) uint32_t get_صفیں_read_اشاریہ_vtcm(
    ثابت خالی * src1,
    uint32_t idx,
    صحیح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    ثابت HVX_سمتیہ v = hvx_vmem((ثابت uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    واپس (uint32_t) Q6_R_vextract_VR(v, off);
}

ساکن خالی get_صفیں_thread_sametype(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت get_صفیں_context * grctx = (ساخت get_صفیں_context *) ڈیٹا;
    ساخت htp_ops_context * octx = grctx->octx;
    ثابت ساخت htp_get_صفیں_kernel_params * kparams = grctx->kparams;
    get_صفیں_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_آغاز + dr * ith;
    اگر (ir0 >= grctx->task_آغاز + grctx->tasks) {
        واپس;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_آغاز + grctx->tasks);
    ثابت uint32_t صف_سائز_بائٹیں = grctx->src0_صف_سائز;
    dma_قطار * dma_q = octx->ctx->dma[ith];

    برائے (uint32_t i = ir0; i < ir1; ++i) {
        ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ثابت uint32_t rem = i - i12 * ne11 * ne10;
        ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ثابت uint32_t i10 = rem - i11 * ne10;

        ثابت uint32_t i01 = grctx->read_اشاریہ(grctx->src1_ڈیٹا, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        ثابت dma_addr_t src0_ڈیٹا = octx->src[0]->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;
        ثابت dma_addr_t dst_ڈیٹا  = octx->dst->ڈیٹا    + i10*nb1  + i11*nb2  + i12*nb3;

        جب (!dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_ڈیٹا, src0_ڈیٹا), nb1, nb01, صف_سائز_بائٹیں, 1)) {
            dma_قطار_نکالو(dma_q);
        }
    }
    dma_قطار_flush(dma_q);
}



ساخت get_صفیں_flat_task {
    dma_addr_t src0_ڈیٹا;
    dma_addr_t dst_ڈیٹا;
};

ساکن اندرونی ساخت get_صفیں_flat_task get_صفیں_flat_calc_task(
    ثابت ساخت htp_ops_context * octx,
    ثابت ساخت get_صفیں_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ثابت ساخت htp_get_صفیں_kernel_params * kparams = grctx->kparams;
    get_صفیں_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;

    ساخت get_صفیں_flat_task task;
    task.src0_ڈیٹا = 0;
    task.dst_ڈیٹا  = octx->dst->ڈیٹا + i10*nb1 + i11*nb2 + i12*nb3;

    اگر (need_src0) {
        ثابت uint32_t i01 = grctx->read_اشاریہ(grctx->src1_ڈیٹا, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        task.src0_ڈیٹا = octx->src[0]->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;
    }

    واپس task;
}

ساکن خالی get_صفیں_thread_flat(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت get_صفیں_context * grctx = (ساخت get_صفیں_context *) ڈیٹا;
    ساخت htp_ops_context * octx = grctx->octx;
    get_صفیں_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_آغاز + dr * ith;
    اگر (ir0 >= grctx->task_آغاز + grctx->tasks) {
        واپس;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_آغاز + grctx->tasks);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت ساخت htp_get_صفیں_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنیاد, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بائٹیں_فی_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنیاد, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بائٹیں_فی_thread;
    ثابت uint32_t src0_بائٹیں = grctx->src0_صف_سائز;
    ثابت uint32_t dst_بائٹیں  = grctx->dst_صف_سائز;
    get_صفیں_flat_compute_fn_t compute = grctx->compute_flat;

    برائے (uint32_t قدم = 0, spad_idx = 0; قدم < ir1 - ir0 && spad_idx < 2; ++قدم, spad_idx++) {
        ثابت uint32_t i = ir0 + قدم;
        ساخت get_صفیں_flat_task task = get_صفیں_flat_calc_task(octx, grctx, i, true);

        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(octx->dst->ڈیٹا,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_سائز),
                       dst_بائٹیں, vtcm_layout->dst_spad_half_سائز, dst_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_سائز, task.src0_ڈیٹا),
                       vtcm_layout->src0_spad_half_سائز, src0_بائٹیں, src0_بائٹیں, 1);
    }

    برائے (uint32_t قدم = 0; قدم < ir1 - ir0; ++قدم) {
        ثابت uint32_t i = ir0 + قدم;
        خالی * dst_spad = (خالی *) dma_قطار_نکالو(dma_q).src;
        خالی * src_spad = (خالی *) dma_قطار_نکالو(dma_q).dst;

        ساخت get_صفیں_flat_task task = get_صفیں_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((اعشاری *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(task.dst_ڈیٹا, dst_spad),
                       dst_بائٹیں, vtcm_layout->dst_spad_half_سائز, dst_بائٹیں, 1);

        ثابت uint32_t اگلا_قدم = قدم + 2;
        اگر (اگلا_قدم < ir1 - ir0) {
            ثابت uint32_t pi = ir0 + اگلا_قدم;
            ساخت get_صفیں_flat_task اگلا_task = get_صفیں_flat_calc_task(octx, grctx, pi, true);
            dma_قطار_ڈالو(dma_q,
                           dma_make_ڈیٹا(src_spad, اگلا_task.src0_ڈیٹا),
                           vtcm_layout->src0_spad_half_سائز, src0_بائٹیں, src0_بائٹیں, 1);
        }
    }

    dma_قطار_flush(dma_q);
}



ساخت get_صفیں_tiled_task {
    dma_addr_t tile_src_بنیاد;
    dma_addr_t dst_ڈیٹا;
    uint32_t   صف;
};

ساکن اندرونی ساخت get_صفیں_tiled_task get_صفیں_tiled_calc_task(
    ثابت ساخت htp_ops_context * octx,
    ثابت ساخت get_صفیں_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_سائز
) {
    ثابت ساخت htp_get_صفیں_kernel_params * kparams = grctx->kparams;
    get_صفیں_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;
    ثابت uint32_t i01 = grctx->read_اشاریہ(grctx->src1_ڈیٹا, i, grctx->idx_type);
    أكد(i01 < ne01);

    ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ثابت uint32_t i02 = i11 - q02 * ne02;
    ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ثابت uint32_t i03 = i12 - q03 * ne03;
    ثابت uint32_t عمود_tile = i01 / HTP_TENSOR_TILE_N_صفیں;
    ثابت uint32_t صف = i01 % HTP_TENSOR_TILE_N_صفیں;
    ثابت dma_addr_t میٹرکس = octx->src[0]->ڈیٹا + i02*nb02 + i03*nb03;

    ساخت get_صفیں_tiled_task task;
    task.tile_src_بنیاد = میٹرکس + (عمود_tile * n_k_tiles) * tile_سائز;
    task.dst_ڈیٹا      = octx->dst->ڈیٹا + i10*nb1 + i11*nb2 + i12*nb3;
    task.صف           = صف;
    واپس task;
}

ساکن خالی get_صفیں_thread_tiled(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت get_صفیں_context * grctx = (ساخت get_صفیں_context *) ڈیٹا;
    ساخت htp_ops_context * octx = grctx->octx;
    get_صفیں_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_آغاز + dr * ith;
    اگر (ir0 >= grctx->task_آغاز + grctx->tasks) {
        واپس;
    }

    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_آغاز + grctx->tasks);
    ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ثابت ساخت htp_get_صفیں_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنیاد, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بائٹیں_فی_thread;
    uint8_t * dst_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنیاد, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بائٹیں_فی_thread;
    dma_قطار * dma_q = octx->ctx->dma[ith];
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t tile_سائز   = grctx->tile_سائز;
    ثابت uint32_t tile_stride = grctx->tile_stride;
    ثابت uint32_t dst_بائٹیں   = grctx->dst_صف_سائز;
    get_صفیں_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    برائے (uint32_t قدم = 0, spad_idx = 0; قدم < ir1 - ir0 && spad_idx < 2; ++قدم, ++spad_idx) {
        ثابت uint32_t i = ir0 + قدم;
        ساخت get_صفیں_tiled_task task = get_صفیں_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_سائز);

        // Dummy writeback to prime the queue with dst descriptor
        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(task.dst_ڈیٹا, dst_spad_بنیاد + spad_idx * vtcm_layout->dst_spad_half_سائز),
                       dst_بائٹیں, vtcm_layout->dst_spad_half_سائز, dst_بائٹیں, 0);

        // Prefetch row tiles
        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(src_spad_بنیاد + spad_idx * vtcm_layout->src0_spad_half_سائز, task.tile_src_بنیاد),
                       tile_stride, tile_سائز, tile_سائز, n_k_tiles);
    }

    برائے (uint32_t قدم = 0; قدم < ir1 - ir0; ++قدم) {
        ثابت uint32_t i = ir0 + قدم;
        اعشاری * dst_spad   = (اعشاری *) dma_قطار_نکالو(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        ساخت get_صفیں_tiled_task task = get_صفیں_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_سائز);

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.صف);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(task.dst_ڈیٹا, dst_spad),
                       dst_بائٹیں, vtcm_layout->dst_spad_half_سائز, dst_بائٹیں, 1);

        ثابت uint32_t اگلا_قدم = قدم + 2;
        اگر (اگلا_قدم < ir1 - ir0) {
            ثابت uint32_t ni = ir0 + اگلا_قدم;
            ساخت get_صفیں_tiled_task اگلا_task = get_صفیں_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_سائز);
            dma_قطار_ڈالو(dma_q,
                           dma_make_ڈیٹا(src_spad, اگلا_task.tile_src_بنیاد),
                           tile_stride, tile_سائز, tile_سائز, n_k_tiles);
        }
    }

    dma_قطار_flush(dma_q);
}

ساکن __attribute__((noinline)) خالی get_صفیں_dma_src1(
    ساخت htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_سائز
) {
    ثابت uint32_t ne10 = octx->src[1]->ne[0];
    ثابت uint32_t ne11 = octx->src[1]->ne[1];
    ثابت uint32_t ne12 = octx->src[1]->ne[2];
    ثابت uint32_t ne13 = octx->src[1]->ne[3];
    ثابت uint32_t nb11 = octx->src[1]->nb[1];
    ثابت uint32_t nb12 = octx->src[1]->nb[2];
    ثابت uint32_t nb13 = octx->src[1]->nb[3];

    dma_قطار * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    برائے (uint32_t i13 = 0; i13 < ne13; ++i13) {
        برائے (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ثابت dma_addr_t src_plane = octx->src[1]->ڈیٹا + i12 * nb12 + i13 * nb13;
            جب (!dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(cur_dst, src_plane),
                                   ne10 * elem_سائز, nb11, ne10 * elem_سائز, ne11)) {
                dma_قطار_نکالو(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_سائز;
        }
    }
    dma_قطار_flush(dma_q);
}

صحیح op_get_صفیں(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_get_صفیں_kernel_params * kparams = (ثابت ساخت htp_get_صفیں_kernel_params *) octx->kernel_params;

    ثابت uint32_t ne00          = octx->src[0]->ne[0];
    ثابت uint32_t src0_صف_سائز = htp_tensor_get_صف_سائز(octx->src[0]->type, ne00);
    اگر (src0_صف_سائز == 0) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (kparams->kernel_type == HTP_GET_صفیں_KERNEL_SAMETYPE) {
        اگر (octx->src[0]->type != octx->dst->type) {
            واپس HTP_حالت_NO_SUPPORT;
        }
    } ورنہ {
        اگر (octx->dst->type != HTP_TYPE_F32) {
            واپس HTP_حالت_NO_SUPPORT;
        }
    }

    اگر (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت ساخت htp_tensor * dst = octx->dst;
    ثابت uint32_t کل_tasks    = kparams->کل_tasks;
    ثابت uint32_t dst_صف_سائز   = (kparams->kernel_type == HTP_GET_صفیں_KERNEL_SAMETYPE) ? src0_صف_سائز : ne00 * حجم(اعشاری);

    uint32_t task_آغاز = 0;
    uint32_t tasks      = کل_tasks;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t tasks_فی_chunk = 1;
        htp_tensor_mdev_صفیں_فی_chunk(dst, dst_صف_سائز / dst->ne[0], dst_صف_سائز, &tasks_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_tasks, tasks_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        task_آغاز = range.آغاز;
        tasks      = range.گنتی;
    }

    اگر (tasks == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, (uint32_t) kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ساخت get_صفیں_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_بنیاد = (uint8_t *)octx->ctx->vtcm_بنیاد;
    grctx.compute_flat = تہی;
    grctx.compute_tiled = تہی;
    grctx.task_آغاز = task_آغاز;
    grctx.tasks = tasks;
    grctx.tasks_فی_thread = octx->ctx->mdev.گنتی == 1 ? kparams->tasks_فی_thread : fastdiv(tasks + n_دھاگے - 1, &octx->n_دھاگے_div);
    grctx.tile_سائز   = 0;
    grctx.tile_stride = 0;
    grctx.src0_صف_سائز = src0_صف_سائز;
    grctx.dst_صف_سائز  = dst_صف_سائز;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_صفیں_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, کل_tasks, grctx.idx_type, n_دھاگے);

    اگر (grctx.vtcm_layout.کل_بائٹیں > octx->ctx->vtcm_سائز) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_سائز, grctx.vtcm_layout.کل_بائٹیں);
        واپس HTP_حالت_INVAL_PARAMS;
    }

    work_قطار_func_t q_func = تہی;
    انتخاب (kparams->kernel_type) {
        حالت HTP_GET_صفیں_KERNEL_SAMETYPE:
            q_func = get_صفیں_thread_sametype;
            توڑو;
        حالت HTP_GET_صفیں_KERNEL_TILED:
            انتخاب (octx->src[0]->type) {
                حالت HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_صف_tiled_f16_f32;  توڑو;
                حالت HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q1_0_f32; توڑو;
                حالت HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q2_0_f32; توڑو;
                حالت HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q8_0_f32; توڑو;
                حالت HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q4_0_f32; توڑو;
                حالت HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_صف_tiled_q4_k_f32; توڑو;
                حالت HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_صف_tiled_q5_k_f32; توڑو;
                حالت HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_صف_tiled_q6_k_f32; توڑو;
                طےشدہ:            واپس HTP_حالت_NO_SUPPORT;
            }
            grctx.tile_سائز   = htp_tensor_get_tile_سائز(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_سائز(octx->src[0]->type);
            q_func = get_صفیں_thread_tiled;
            توڑو;
        حالت HTP_GET_صفیں_KERNEL_FLAT:
            انتخاب (octx->src[0]->type) {
                حالت HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_صف_flat_f16_f32;  توڑو;
                حالت HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_صف_flat_bf16_f32; توڑو;
                حالت HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_صف_flat_q1_0_f32; توڑو;
                حالت HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_صف_flat_q2_0_f32; توڑو;
                حالت HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_صف_flat_q8_0_f32; توڑو;
                حالت HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_صف_flat_q4_0_f32; توڑو;
                حالت HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_صف_flat_q4_k_f32; توڑو;
                حالت HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_صف_flat_q5_k_f32; توڑو;
                حالت HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_صف_flat_q6_k_f32; توڑو;
                طےشدہ:            واپس HTP_حالت_NO_SUPPORT;
            }
            q_func = get_صفیں_thread_flat;
            توڑو;
        طےشدہ:
            واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t elem_سائز = (grctx.idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_بائٹیں  = (size_t) کل_tasks * elem_سائز;

    اگر (!htp_tensor_is_extended(octx->src[1]) && src1_بائٹیں <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_سائز)) {
        grctx.src1_ڈیٹا  = (ثابت خالی *) (uintptr_t) octx->src[1]->ڈیٹا;
        grctx.read_اشاریہ = get_صفیں_read_اشاریہ_ddr;
    } ورنہ {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_بنیاد, grctx.vtcm_layout.off_src1);
        get_صفیں_dma_src1(octx, vtcm_src1, elem_سائز);
        grctx.src1_ڈیٹا  = vtcm_src1;
        grctx.read_اشاریہ = get_صفیں_read_اشاریہ_vtcm;
    }

    FARF(بلند_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_بائٹیں_فی_thread * n_دھاگے,
         grctx.vtcm_layout.dst_بائٹیں_فی_thread  * n_دھاگے,
         kparams->kernel_type, n_دھاگے);

    work_قطار_تسلسل(octx->ctx->work_قطار, q_func, &grctx, n_دھاگے);
    واپس HTP_حالت_OK;
}
