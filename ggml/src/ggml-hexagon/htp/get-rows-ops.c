#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "dma-queue.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-utils.h"
#ضمن "hvx-quant.h"
#ضمن "get-rows-ops.h"
#ضمن "work-queue.h"

بنية get_صفون_context;

تعريفنوع uint32_t (*get_صفون_read_اشاريو_fn_t)(ثابت فراغ * src1, uint32_t idx, صحيح idx_type);
تعريفنوع فراغ (*get_صفون_flat_compute_fn_t)(عائم * dst_spad, ثابت فراغ * src_spad, uint32_t cur_elems);
تعريفنوع فراغ (*get_صفون_tiled_compute_fn_t)(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف);

بنية get_صفون_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_get_صفون_kernel_params * kparams;
    بنية htp_get_صفون_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_بنياد;
    ثابت فراغ * src1_معلومات;
    get_صفون_read_اشاريو_fn_t    read_اشاريو;
    get_صفون_flat_compute_fn_t  compute_flat;
    get_صفون_tiled_compute_fn_t compute_tiled;
    uint32_t task_شروعات;
    uint32_t tasks;
    uint32_t tasks_في_thread;
    uint32_t tile_سائيز;
    uint32_t tile_stride;
    uint32_t src0_صف_سائيز;
    uint32_t dst_صف_سائيز;
    صحيح idx_type;
};

#وصف_ڪريو get_صفون_preamble                      \
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

ساكن uint32_t get_صفون_read_اشاريو_ddr(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    أرجع *(ثابت uint32_t *) ((ثابت uint8_t *) src1 + off);
}

ساكن __attribute__((noinline)) uint32_t get_صفون_read_اشاريو_vtcm(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    ثابت HVX_ويڪٽر v = hvx_vmem((ثابت uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    أرجع (uint32_t) Q6_R_vextract_VR(v, off);
}

ساكن فراغ get_صفون_thread_sametype(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية get_صفون_context * grctx = (بنية get_صفون_context *) معلومات;
    بنية htp_ops_context * octx = grctx->octx;
    ثابت بنية htp_get_صفون_kernel_params * kparams = grctx->kparams;
    get_صفون_preamble;

    ثابت uint32_t dr  = grctx->tasks_في_thread;
    ثابت uint32_t ir0 = grctx->task_شروعات + dr * ith;
    إذا (ir0 >= grctx->task_شروعات + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروعات + grctx->tasks);
    ثابت uint32_t صف_سائيز_بائيٽون = grctx->src0_صف_سائيز;
    dma_قطار * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t i = ir0; i < ir1; ++i) {
        ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ثابت uint32_t rem = i - i12 * ne11 * ne10;
        ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ثابت uint32_t i10 = rem - i11 * ne10;

        ثابت uint32_t i01 = grctx->read_اشاريو(grctx->src1_معلومات, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        ثابت dma_addr_t src0_معلومات = octx->src[0]->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
        ثابت dma_addr_t dst_معلومات  = octx->dst->معلومات    + i10*nb1  + i11*nb2  + i12*nb3;

        طالما (!dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_معلومات, src0_معلومات), nb1, nb01, صف_سائيز_بائيٽون, 1)) {
            dma_قطار_ڪڍو(dma_q);
        }
    }
    dma_قطار_flush(dma_q);
}



بنية get_صفون_flat_task {
    dma_addr_t src0_معلومات;
    dma_addr_t dst_معلومات;
};

ساكن مضمن بنية get_صفون_flat_task get_صفون_flat_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_صفون_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ثابت بنية htp_get_صفون_kernel_params * kparams = grctx->kparams;
    get_صفون_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;

    بنية get_صفون_flat_task task;
    task.src0_معلومات = 0;
    task.dst_معلومات  = octx->dst->معلومات + i10*nb1 + i11*nb2 + i12*nb3;

    إذا (need_src0) {
        ثابت uint32_t i01 = grctx->read_اشاريو(grctx->src1_معلومات, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        task.src0_معلومات = octx->src[0]->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
    }

    أرجع task;
}

ساكن فراغ get_صفون_thread_flat(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية get_صفون_context * grctx = (بنية get_صفون_context *) معلومات;
    بنية htp_ops_context * octx = grctx->octx;
    get_صفون_preamble;

    ثابت uint32_t dr  = grctx->tasks_في_thread;
    ثابت uint32_t ir0 = grctx->task_شروعات + dr * ith;
    إذا (ir0 >= grctx->task_شروعات + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروعات + grctx->tasks);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت بنية htp_get_صفون_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنياد, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بائيٽون_في_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنياد, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بائيٽون_في_thread;
    ثابت uint32_t src0_بائيٽون = grctx->src0_صف_سائيز;
    ثابت uint32_t dst_بائيٽون  = grctx->dst_صف_سائيز;
    get_صفون_flat_compute_fn_t compute = grctx->compute_flat;

    لكل (uint32_t قدم = 0, spad_idx = 0; قدم < ir1 - ir0 && spad_idx < 2; ++قدم, spad_idx++) {
        ثابت uint32_t i = ir0 + قدم;
        بنية get_صفون_flat_task task = get_صفون_flat_calc_task(octx, grctx, i, true);

        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(octx->dst->معلومات,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_سائيز),
                       dst_بائيٽون, vtcm_layout->dst_spad_half_سائيز, dst_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_سائيز, task.src0_معلومات),
                       vtcm_layout->src0_spad_half_سائيز, src0_بائيٽون, src0_بائيٽون, 1);
    }

    لكل (uint32_t قدم = 0; قدم < ir1 - ir0; ++قدم) {
        ثابت uint32_t i = ir0 + قدم;
        فراغ * dst_spad = (فراغ *) dma_قطار_ڪڍو(dma_q).src;
        فراغ * src_spad = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;

        بنية get_صفون_flat_task task = get_صفون_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((عائم *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(task.dst_معلومات, dst_spad),
                       dst_بائيٽون, vtcm_layout->dst_spad_half_سائيز, dst_بائيٽون, 1);

        ثابت uint32_t ايندڙ_قدم = قدم + 2;
        إذا (ايندڙ_قدم < ir1 - ir0) {
            ثابت uint32_t pi = ir0 + ايندڙ_قدم;
            بنية get_صفون_flat_task ايندڙ_task = get_صفون_flat_calc_task(octx, grctx, pi, true);
            dma_قطار_ڌڪو(dma_q,
                           dma_make_معلومات(src_spad, ايندڙ_task.src0_معلومات),
                           vtcm_layout->src0_spad_half_سائيز, src0_بائيٽون, src0_بائيٽون, 1);
        }
    }

    dma_قطار_flush(dma_q);
}



بنية get_صفون_tiled_task {
    dma_addr_t tile_src_بنياد;
    dma_addr_t dst_معلومات;
    uint32_t   صف;
};

ساكن مضمن بنية get_صفون_tiled_task get_صفون_tiled_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_صفون_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_سائيز
) {
    ثابت بنية htp_get_صفون_kernel_params * kparams = grctx->kparams;
    get_صفون_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;
    ثابت uint32_t i01 = grctx->read_اشاريو(grctx->src1_معلومات, i, grctx->idx_type);
    أكد(i01 < ne01);

    ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ثابت uint32_t i02 = i11 - q02 * ne02;
    ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ثابت uint32_t i03 = i12 - q03 * ne03;
    ثابت uint32_t ڪالم_tile = i01 / HTP_TENSOR_TILE_N_صفون;
    ثابت uint32_t صف = i01 % HTP_TENSOR_TILE_N_صفون;
    ثابت dma_addr_t ميٽرڪس = octx->src[0]->معلومات + i02*nb02 + i03*nb03;

    بنية get_صفون_tiled_task task;
    task.tile_src_بنياد = ميٽرڪس + (ڪالم_tile * n_k_tiles) * tile_سائيز;
    task.dst_معلومات      = octx->dst->معلومات + i10*nb1 + i11*nb2 + i12*nb3;
    task.صف           = صف;
    أرجع task;
}

ساكن فراغ get_صفون_thread_tiled(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية get_صفون_context * grctx = (بنية get_صفون_context *) معلومات;
    بنية htp_ops_context * octx = grctx->octx;
    get_صفون_preamble;

    ثابت uint32_t dr  = grctx->tasks_في_thread;
    ثابت uint32_t ir0 = grctx->task_شروعات + dr * ith;
    إذا (ir0 >= grctx->task_شروعات + grctx->tasks) {
        أرجع;
    }

    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروعات + grctx->tasks);
    ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ثابت بنية htp_get_صفون_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنياد, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بائيٽون_في_thread;
    uint8_t * dst_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنياد, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بائيٽون_في_thread;
    dma_قطار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t tile_سائيز   = grctx->tile_سائيز;
    ثابت uint32_t tile_stride = grctx->tile_stride;
    ثابت uint32_t dst_بائيٽون   = grctx->dst_صف_سائيز;
    get_صفون_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    لكل (uint32_t قدم = 0, spad_idx = 0; قدم < ir1 - ir0 && spad_idx < 2; ++قدم, ++spad_idx) {
        ثابت uint32_t i = ir0 + قدم;
        بنية get_صفون_tiled_task task = get_صفون_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_سائيز);

        // Dummy writeback to prime the queue with dst descriptor
        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(task.dst_معلومات, dst_spad_بنياد + spad_idx * vtcm_layout->dst_spad_half_سائيز),
                       dst_بائيٽون, vtcm_layout->dst_spad_half_سائيز, dst_بائيٽون, 0);

        // Prefetch row tiles
        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(src_spad_بنياد + spad_idx * vtcm_layout->src0_spad_half_سائيز, task.tile_src_بنياد),
                       tile_stride, tile_سائيز, tile_سائيز, n_k_tiles);
    }

    لكل (uint32_t قدم = 0; قدم < ir1 - ir0; ++قدم) {
        ثابت uint32_t i = ir0 + قدم;
        عائم * dst_spad   = (عائم *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        بنية get_صفون_tiled_task task = get_صفون_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_سائيز);

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.صف);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(task.dst_معلومات, dst_spad),
                       dst_بائيٽون, vtcm_layout->dst_spad_half_سائيز, dst_بائيٽون, 1);

        ثابت uint32_t ايندڙ_قدم = قدم + 2;
        إذا (ايندڙ_قدم < ir1 - ir0) {
            ثابت uint32_t ni = ir0 + ايندڙ_قدم;
            بنية get_صفون_tiled_task ايندڙ_task = get_صفون_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_سائيز);
            dma_قطار_ڌڪو(dma_q,
                           dma_make_معلومات(src_spad, ايندڙ_task.tile_src_بنياد),
                           tile_stride, tile_سائيز, tile_سائيز, n_k_tiles);
        }
    }

    dma_قطار_flush(dma_q);
}

ساكن __attribute__((noinline)) فراغ get_صفون_dma_src1(
    بنية htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_سائيز
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
    لكل (uint32_t i13 = 0; i13 < ne13; ++i13) {
        لكل (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ثابت dma_addr_t src_plane = octx->src[1]->معلومات + i12 * nb12 + i13 * nb13;
            طالما (!dma_قطار_ڌڪو(dma_q, dma_make_معلومات(cur_dst, src_plane),
                                   ne10 * elem_سائيز, nb11, ne10 * elem_سائيز, ne11)) {
                dma_قطار_ڪڍو(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_سائيز;
        }
    }
    dma_قطار_flush(dma_q);
}

صحيح op_get_صفون(بنية htp_ops_context * octx) {
    ثابت بنية htp_get_صفون_kernel_params * kparams = (ثابت بنية htp_get_صفون_kernel_params *) octx->kernel_params;

    ثابت uint32_t ne00          = octx->src[0]->ne[0];
    ثابت uint32_t src0_صف_سائيز = htp_tensor_get_صف_سائيز(octx->src[0]->type, ne00);
    إذا (src0_صف_سائيز == 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (kparams->kernel_type == HTP_GET_صفون_KERNEL_SAMETYPE) {
        إذا (octx->src[0]->type != octx->dst->type) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
    } وإلا {
        إذا (octx->dst->type != HTP_TYPE_F32) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
    }

    إذا (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * dst = octx->dst;
    ثابت uint32_t ڪل_tasks    = kparams->ڪل_tasks;
    ثابت uint32_t dst_صف_سائيز   = (kparams->kernel_type == HTP_GET_صفون_KERNEL_SAMETYPE) ? src0_صف_سائيز : ne00 * حجم(عائم);

    uint32_t task_شروعات = 0;
    uint32_t tasks      = ڪل_tasks;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t tasks_في_chunk = 1;
        htp_tensor_mdev_صفون_في_chunk(dst, dst_صف_سائيز / dst->ne[0], dst_صف_سائيز, &tasks_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_tasks, tasks_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        task_شروعات = range.شروعات;
        tasks      = range.ڳڻپ;
    }

    إذا (tasks == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, (uint32_t) kparams->n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    بنية get_صفون_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_بنياد = (uint8_t *)octx->ctx->vtcm_بنياد;
    grctx.compute_flat = عدم;
    grctx.compute_tiled = عدم;
    grctx.task_شروعات = task_شروعات;
    grctx.tasks = tasks;
    grctx.tasks_في_thread = octx->ctx->mdev.ڳڻپ == 1 ? kparams->tasks_في_thread : fastdiv(tasks + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);
    grctx.tile_سائيز   = 0;
    grctx.tile_stride = 0;
    grctx.src0_صف_سائيز = src0_صف_سائيز;
    grctx.dst_صف_سائيز  = dst_صف_سائيز;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_صفون_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, ڪل_tasks, grctx.idx_type, n_ڌاڳا);

    إذا (grctx.vtcm_layout.ڪل_بائيٽون > octx->ctx->vtcm_سائيز) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_سائيز, grctx.vtcm_layout.ڪل_بائيٽون);
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    work_قطار_func_t q_func = عدم;
    بدل (kparams->kernel_type) {
        حالة HTP_GET_صفون_KERNEL_SAMETYPE:
            q_func = get_صفون_thread_sametype;
            اكسر;
        حالة HTP_GET_صفون_KERNEL_TILED:
            بدل (octx->src[0]->type) {
                حالة HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_صف_tiled_f16_f32;  اكسر;
                حالة HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q1_0_f32; اكسر;
                حالة HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q2_0_f32; اكسر;
                حالة HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q8_0_f32; اكسر;
                حالة HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_صف_tiled_q4_0_f32; اكسر;
                حالة HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_صف_tiled_q4_k_f32; اكسر;
                حالة HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_صف_tiled_q5_k_f32; اكسر;
                حالة HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_صف_tiled_q6_k_f32; اكسر;
                افتراضي:            أرجع HTP_حالت_NO_SUPPORT;
            }
            grctx.tile_سائيز   = htp_tensor_get_tile_سائيز(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_سائيز(octx->src[0]->type);
            q_func = get_صفون_thread_tiled;
            اكسر;
        حالة HTP_GET_صفون_KERNEL_FLAT:
            بدل (octx->src[0]->type) {
                حالة HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_صف_flat_f16_f32;  اكسر;
                حالة HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_صف_flat_bf16_f32; اكسر;
                حالة HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_صف_flat_q1_0_f32; اكسر;
                حالة HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_صف_flat_q2_0_f32; اكسر;
                حالة HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_صف_flat_q8_0_f32; اكسر;
                حالة HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_صف_flat_q4_0_f32; اكسر;
                حالة HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_صف_flat_q4_k_f32; اكسر;
                حالة HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_صف_flat_q5_k_f32; اكسر;
                حالة HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_صف_flat_q6_k_f32; اكسر;
                افتراضي:            أرجع HTP_حالت_NO_SUPPORT;
            }
            q_func = get_صفون_thread_flat;
            اكسر;
        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t elem_سائيز = (grctx.idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_بائيٽون  = (size_t) ڪل_tasks * elem_سائيز;

    إذا (!htp_tensor_is_extended(octx->src[1]) && src1_بائيٽون <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_سائيز)) {
        grctx.src1_معلومات  = (ثابت فراغ *) (uintptr_t) octx->src[1]->معلومات;
        grctx.read_اشاريو = get_صفون_read_اشاريو_ddr;
    } وإلا {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_بنياد, grctx.vtcm_layout.off_src1);
        get_صفون_dma_src1(octx, vtcm_src1, elem_سائيز);
        grctx.src1_معلومات  = vtcm_src1;
        grctx.read_اشاريو = get_صفون_read_اشاريو_vtcm;
    }

    FARF(مٿاهون_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_بائيٽون_في_thread * n_ڌاڳا,
         grctx.vtcm_layout.dst_بائيٽون_في_thread  * n_ڌاڳا,
         kparams->kernel_type, n_ڌاڳا);

    work_قطار_لاڳيتو(octx->ctx->work_قطار, q_func, &grctx, n_ڌاڳا);
    أرجع HTP_حالت_OK;
}
