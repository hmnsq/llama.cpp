#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#تعریف_کن GGML_COMMON_DECL_C
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

بنية get_ردیفها_context;

تعريفنوع uint32_t (*get_ردیفها_read_شاخص_fn_t)(ثابت فراغ * src1, uint32_t idx, صحيح idx_type);
تعريفنوع فراغ (*get_ردیفها_flat_compute_fn_t)(عائم * dst_spad, ثابت فراغ * src_spad, uint32_t cur_elems);
تعريفنوع فراغ (*get_ردیفها_tiled_compute_fn_t)(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ردیف);

بنية get_ردیفها_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_get_ردیفها_kernel_params * kparams;
    بنية htp_get_ردیفها_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_پایه;
    ثابت فراغ * src1_داده;
    get_ردیفها_read_شاخص_fn_t    read_شاخص;
    get_ردیفها_flat_compute_fn_t  compute_flat;
    get_ردیفها_tiled_compute_fn_t compute_tiled;
    uint32_t task_شروع;
    uint32_t tasks;
    uint32_t tasks_در_هر_thread;
    uint32_t tile_اندازه;
    uint32_t tile_stride;
    uint32_t src0_ردیف_اندازه;
    uint32_t dst_ردیف_اندازه;
    صحيح idx_type;
};

#تعریف_کن get_ردیفها_preamble                      \
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

ساكن uint32_t get_ردیفها_read_شاخص_ddr(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    أرجع *(ثابت uint32_t *) ((ثابت uint8_t *) src1 + off);
}

ساكن __attribute__((noinline)) uint32_t get_ردیفها_read_شاخص_vtcm(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    ثابت HVX_وکتور v = hvx_vmem((ثابت uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    أرجع (uint32_t) Q6_R_vextract_VR(v, off);
}

ساكن فراغ get_ردیفها_thread_sametype(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية get_ردیفها_context * grctx = (بنية get_ردیفها_context *) داده;
    بنية htp_ops_context * octx = grctx->octx;
    ثابت بنية htp_get_ردیفها_kernel_params * kparams = grctx->kparams;
    get_ردیفها_preamble;

    ثابت uint32_t dr  = grctx->tasks_در_هر_thread;
    ثابت uint32_t ir0 = grctx->task_شروع + dr * ith;
    إذا (ir0 >= grctx->task_شروع + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروع + grctx->tasks);
    ثابت uint32_t ردیف_اندازه_بایتها = grctx->src0_ردیف_اندازه;
    dma_صف * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t i = ir0; i < ir1; ++i) {
        ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ثابت uint32_t rem = i - i12 * ne11 * ne10;
        ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ثابت uint32_t i10 = rem - i11 * ne10;

        ثابت uint32_t i01 = grctx->read_شاخص(grctx->src1_داده, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        ثابت dma_addr_t src0_داده = octx->src[0]->داده + i01*nb01 + i02*nb02 + i03*nb03;
        ثابت dma_addr_t dst_داده  = octx->dst->داده    + i10*nb1  + i11*nb2  + i12*nb3;

        طالما (!dma_صف_افزودن(dma_q, dma_make_داده(dst_داده, src0_داده), nb1, nb01, ردیف_اندازه_بایتها, 1)) {
            dma_صف_برداشت(dma_q);
        }
    }
    dma_صف_flush(dma_q);
}



بنية get_ردیفها_flat_task {
    dma_addr_t src0_داده;
    dma_addr_t dst_داده;
};

ساكن مضمن بنية get_ردیفها_flat_task get_ردیفها_flat_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_ردیفها_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ثابت بنية htp_get_ردیفها_kernel_params * kparams = grctx->kparams;
    get_ردیفها_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;

    بنية get_ردیفها_flat_task task;
    task.src0_داده = 0;
    task.dst_داده  = octx->dst->داده + i10*nb1 + i11*nb2 + i12*nb3;

    إذا (need_src0) {
        ثابت uint32_t i01 = grctx->read_شاخص(grctx->src1_داده, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        task.src0_داده = octx->src[0]->داده + i01*nb01 + i02*nb02 + i03*nb03;
    }

    أرجع task;
}

ساكن فراغ get_ردیفها_thread_flat(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية get_ردیفها_context * grctx = (بنية get_ردیفها_context *) داده;
    بنية htp_ops_context * octx = grctx->octx;
    get_ردیفها_preamble;

    ثابت uint32_t dr  = grctx->tasks_در_هر_thread;
    ثابت uint32_t ir0 = grctx->task_شروع + dr * ith;
    إذا (ir0 >= grctx->task_شروع + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروع + grctx->tasks);

    dma_صف * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت بنية htp_get_ردیفها_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_پایه, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بایتها_در_هر_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_پایه, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بایتها_در_هر_thread;
    ثابت uint32_t src0_بایتها = grctx->src0_ردیف_اندازه;
    ثابت uint32_t dst_بایتها  = grctx->dst_ردیف_اندازه;
    get_ردیفها_flat_compute_fn_t compute = grctx->compute_flat;

    لكل (uint32_t گام = 0, spad_idx = 0; گام < ir1 - ir0 && spad_idx < 2; ++گام, spad_idx++) {
        ثابت uint32_t i = ir0 + گام;
        بنية get_ردیفها_flat_task task = get_ردیفها_flat_calc_task(octx, grctx, i, true);

        dma_صف_افزودن(dma_q,
                       dma_make_داده(octx->dst->داده,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_اندازه),
                       dst_بایتها, vtcm_layout->dst_spad_half_اندازه, dst_بایتها, 0);
        dma_صف_افزودن(dma_q,
                       dma_make_داده(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_اندازه, task.src0_داده),
                       vtcm_layout->src0_spad_half_اندازه, src0_بایتها, src0_بایتها, 1);
    }

    لكل (uint32_t گام = 0; گام < ir1 - ir0; ++گام) {
        ثابت uint32_t i = ir0 + گام;
        فراغ * dst_spad = (فراغ *) dma_صف_برداشت(dma_q).src;
        فراغ * src_spad = (فراغ *) dma_صف_برداشت(dma_q).dst;

        بنية get_ردیفها_flat_task task = get_ردیفها_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((عائم *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_صف_افزودن(dma_q,
                       dma_make_داده(task.dst_داده, dst_spad),
                       dst_بایتها, vtcm_layout->dst_spad_half_اندازه, dst_بایتها, 1);

        ثابت uint32_t بعدی_گام = گام + 2;
        إذا (بعدی_گام < ir1 - ir0) {
            ثابت uint32_t pi = ir0 + بعدی_گام;
            بنية get_ردیفها_flat_task بعدی_task = get_ردیفها_flat_calc_task(octx, grctx, pi, true);
            dma_صف_افزودن(dma_q,
                           dma_make_داده(src_spad, بعدی_task.src0_داده),
                           vtcm_layout->src0_spad_half_اندازه, src0_بایتها, src0_بایتها, 1);
        }
    }

    dma_صف_flush(dma_q);
}



بنية get_ردیفها_tiled_task {
    dma_addr_t tile_src_پایه;
    dma_addr_t dst_داده;
    uint32_t   ردیف;
};

ساكن مضمن بنية get_ردیفها_tiled_task get_ردیفها_tiled_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_ردیفها_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_اندازه
) {
    ثابت بنية htp_get_ردیفها_kernel_params * kparams = grctx->kparams;
    get_ردیفها_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;
    ثابت uint32_t i01 = grctx->read_شاخص(grctx->src1_داده, i, grctx->idx_type);
    أكد(i01 < ne01);

    ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ثابت uint32_t i02 = i11 - q02 * ne02;
    ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ثابت uint32_t i03 = i12 - q03 * ne03;
    ثابت uint32_t ستونی_tile = i01 / HTP_TENSOR_TILE_N_ردیفها;
    ثابت uint32_t ردیف = i01 % HTP_TENSOR_TILE_N_ردیفها;
    ثابت dma_addr_t ماتریس = octx->src[0]->داده + i02*nb02 + i03*nb03;

    بنية get_ردیفها_tiled_task task;
    task.tile_src_پایه = ماتریس + (ستونی_tile * n_k_tiles) * tile_اندازه;
    task.dst_داده      = octx->dst->داده + i10*nb1 + i11*nb2 + i12*nb3;
    task.ردیف           = ردیف;
    أرجع task;
}

ساكن فراغ get_ردیفها_thread_tiled(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية get_ردیفها_context * grctx = (بنية get_ردیفها_context *) داده;
    بنية htp_ops_context * octx = grctx->octx;
    get_ردیفها_preamble;

    ثابت uint32_t dr  = grctx->tasks_در_هر_thread;
    ثابت uint32_t ir0 = grctx->task_شروع + dr * ith;
    إذا (ir0 >= grctx->task_شروع + grctx->tasks) {
        أرجع;
    }

    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروع + grctx->tasks);
    ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ثابت بنية htp_get_ردیفها_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_پایه, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بایتها_در_هر_thread;
    uint8_t * dst_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_پایه, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بایتها_در_هر_thread;
    dma_صف * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t tile_اندازه   = grctx->tile_اندازه;
    ثابت uint32_t tile_stride = grctx->tile_stride;
    ثابت uint32_t dst_بایتها   = grctx->dst_ردیف_اندازه;
    get_ردیفها_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    لكل (uint32_t گام = 0, spad_idx = 0; گام < ir1 - ir0 && spad_idx < 2; ++گام, ++spad_idx) {
        ثابت uint32_t i = ir0 + گام;
        بنية get_ردیفها_tiled_task task = get_ردیفها_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_اندازه);

        // Dummy writeback to prime the queue with dst descriptor
        dma_صف_افزودن(dma_q,
                       dma_make_داده(task.dst_داده, dst_spad_پایه + spad_idx * vtcm_layout->dst_spad_half_اندازه),
                       dst_بایتها, vtcm_layout->dst_spad_half_اندازه, dst_بایتها, 0);

        // Prefetch row tiles
        dma_صف_افزودن(dma_q,
                       dma_make_داده(src_spad_پایه + spad_idx * vtcm_layout->src0_spad_half_اندازه, task.tile_src_پایه),
                       tile_stride, tile_اندازه, tile_اندازه, n_k_tiles);
    }

    لكل (uint32_t گام = 0; گام < ir1 - ir0; ++گام) {
        ثابت uint32_t i = ir0 + گام;
        عائم * dst_spad   = (عائم *) dma_صف_برداشت(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        بنية get_ردیفها_tiled_task task = get_ردیفها_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_اندازه);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.ردیف);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_صف_افزودن(dma_q,
                       dma_make_داده(task.dst_داده, dst_spad),
                       dst_بایتها, vtcm_layout->dst_spad_half_اندازه, dst_بایتها, 1);

        ثابت uint32_t بعدی_گام = گام + 2;
        إذا (بعدی_گام < ir1 - ir0) {
            ثابت uint32_t ni = ir0 + بعدی_گام;
            بنية get_ردیفها_tiled_task بعدی_task = get_ردیفها_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_اندازه);
            dma_صف_افزودن(dma_q,
                           dma_make_داده(src_spad, بعدی_task.tile_src_پایه),
                           tile_stride, tile_اندازه, tile_اندازه, n_k_tiles);
        }
    }

    dma_صف_flush(dma_q);
}

ساكن __attribute__((noinline)) فراغ get_ردیفها_dma_src1(
    بنية htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_اندازه
) {
    ثابت uint32_t ne10 = octx->src[1]->ne[0];
    ثابت uint32_t ne11 = octx->src[1]->ne[1];
    ثابت uint32_t ne12 = octx->src[1]->ne[2];
    ثابت uint32_t ne13 = octx->src[1]->ne[3];
    ثابت uint32_t nb11 = octx->src[1]->nb[1];
    ثابت uint32_t nb12 = octx->src[1]->nb[2];
    ثابت uint32_t nb13 = octx->src[1]->nb[3];

    dma_صف * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    لكل (uint32_t i13 = 0; i13 < ne13; ++i13) {
        لكل (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ثابت dma_addr_t src_plane = octx->src[1]->داده + i12 * nb12 + i13 * nb13;
            طالما (!dma_صف_افزودن(dma_q, dma_make_داده(cur_dst, src_plane),
                                   ne10 * elem_اندازه, nb11, ne10 * elem_اندازه, ne11)) {
                dma_صف_برداشت(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_اندازه;
        }
    }
    dma_صف_flush(dma_q);
}

صحيح op_get_ردیفها(بنية htp_ops_context * octx) {
    ثابت بنية htp_get_ردیفها_kernel_params * kparams = (ثابت بنية htp_get_ردیفها_kernel_params *) octx->kernel_params;

    ثابت uint32_t ne00          = octx->src[0]->ne[0];
    ثابت uint32_t src0_ردیف_اندازه = htp_tensor_get_ردیف_اندازه(octx->src[0]->type, ne00);
    إذا (src0_ردیف_اندازه == 0) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (kparams->kernel_type == HTP_GET_ردیفها_KERNEL_SAMETYPE) {
        إذا (octx->src[0]->type != octx->dst->type) {
            أرجع HTP_وضعیت_NO_SUPPORT;
        }
    } وإلا {
        إذا (octx->dst->type != HTP_TYPE_F32) {
            أرجع HTP_وضعیت_NO_SUPPORT;
        }
    }

    إذا (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * dst = octx->dst;
    ثابت uint32_t مجموع_tasks    = kparams->مجموع_tasks;
    ثابت uint32_t dst_ردیف_اندازه   = (kparams->kernel_type == HTP_GET_ردیفها_KERNEL_SAMETYPE) ? src0_ردیف_اندازه : ne00 * حجم(عائم);

    uint32_t task_شروع = 0;
    uint32_t tasks      = مجموع_tasks;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t tasks_در_هر_chunk = 1;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, dst_ردیف_اندازه / dst->ne[0], dst_ردیف_اندازه, &tasks_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_tasks, tasks_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        task_شروع = range.شروع;
        tasks      = range.شمار;
    }

    إذا (tasks == 0) {
        أرجع HTP_وضعیت_OK;
    }

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, (uint32_t) kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    بنية get_ردیفها_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_پایه = (uint8_t *)octx->ctx->vtcm_پایه;
    grctx.compute_flat = عدم;
    grctx.compute_tiled = عدم;
    grctx.task_شروع = task_شروع;
    grctx.tasks = tasks;
    grctx.tasks_در_هر_thread = octx->ctx->mdev.شمار == 1 ? kparams->tasks_در_هر_thread : fastdiv(tasks + n_نخها - 1, &octx->n_نخها_div);
    grctx.tile_اندازه   = 0;
    grctx.tile_stride = 0;
    grctx.src0_ردیف_اندازه = src0_ردیف_اندازه;
    grctx.dst_ردیف_اندازه  = dst_ردیف_اندازه;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_ردیفها_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, مجموع_tasks, grctx.idx_type, n_نخها);

    إذا (grctx.vtcm_layout.مجموع_بایتها > octx->ctx->vtcm_اندازه) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_اندازه, grctx.vtcm_layout.مجموع_بایتها);
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    work_صف_func_t q_func = عدم;
    بدل (kparams->kernel_type) {
        حالة HTP_GET_ردیفها_KERNEL_SAMETYPE:
            q_func = get_ردیفها_thread_sametype;
            اكسر;
        حالة HTP_GET_ردیفها_KERNEL_TILED:
            بدل (octx->src[0]->type) {
                حالة HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_ردیف_tiled_f16_f32;  اكسر;
                حالة HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_ردیف_tiled_q1_0_f32; اكسر;
                حالة HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_ردیف_tiled_q2_0_f32; اكسر;
                حالة HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_ردیف_tiled_q8_0_f32; اكسر;
                حالة HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_ردیف_tiled_q4_0_f32; اكسر;
                حالة HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_ردیف_tiled_q4_k_f32; اكسر;
                حالة HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_ردیف_tiled_q5_k_f32; اكسر;
                حالة HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_ردیف_tiled_q6_k_f32; اكسر;
                افتراضي:            أرجع HTP_وضعیت_NO_SUPPORT;
            }
            grctx.tile_اندازه   = htp_tensor_get_tile_اندازه(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_اندازه(octx->src[0]->type);
            q_func = get_ردیفها_thread_tiled;
            اكسر;
        حالة HTP_GET_ردیفها_KERNEL_FLAT:
            بدل (octx->src[0]->type) {
                حالة HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_ردیف_flat_f16_f32;  اكسر;
                حالة HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_ردیف_flat_bf16_f32; اكسر;
                حالة HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_ردیف_flat_q1_0_f32; اكسر;
                حالة HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_ردیف_flat_q2_0_f32; اكسر;
                حالة HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_ردیف_flat_q8_0_f32; اكسر;
                حالة HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_ردیف_flat_q4_0_f32; اكسر;
                حالة HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_ردیف_flat_q4_k_f32; اكسر;
                حالة HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_ردیف_flat_q5_k_f32; اكسر;
                حالة HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_ردیف_flat_q6_k_f32; اكسر;
                افتراضي:            أرجع HTP_وضعیت_NO_SUPPORT;
            }
            q_func = get_ردیفها_thread_flat;
            اكسر;
        افتراضي:
            أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t elem_اندازه = (grctx.idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_بایتها  = (size_t) مجموع_tasks * elem_اندازه;

    إذا (!htp_tensor_is_extended(octx->src[1]) && src1_بایتها <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_اندازه)) {
        grctx.src1_داده  = (ثابت فراغ *) (uintptr_t) octx->src[1]->داده;
        grctx.read_شاخص = get_ردیفها_read_شاخص_ddr;
    } وإلا {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_پایه, grctx.vtcm_layout.off_src1);
        get_ردیفها_dma_src1(octx, vtcm_src1, elem_اندازه);
        grctx.src1_داده  = vtcm_src1;
        grctx.read_شاخص = get_ردیفها_read_شاخص_vtcm;
    }

    FARF(بالا_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_بایتها_در_هر_thread * n_نخها,
         grctx.vtcm_layout.dst_بایتها_در_هر_thread  * n_نخها,
         kparams->kernel_type, n_نخها);

    work_صف_پیاپی(octx->ctx->work_صف, q_func, &grctx, n_نخها);
    أرجع HTP_وضعیت_OK;
}
