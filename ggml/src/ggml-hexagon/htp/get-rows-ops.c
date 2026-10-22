#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#تعریف_کړه GGML_COMMON_DECL_C
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

بنية get_لیکې_context;

تعريفنوع uint32_t (*get_لیکې_read_شاخص_fn_t)(ثابت فراغ * src1, uint32_t idx, صحيح idx_type);
تعريفنوع فراغ (*get_لیکې_flat_compute_fn_t)(عائم * dst_spad, ثابت فراغ * src_spad, uint32_t cur_elems);
تعريفنوع فراغ (*get_لیکې_tiled_compute_fn_t)(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t لیکه);

بنية get_لیکې_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_get_لیکې_kernel_params * kparams;
    بنية htp_get_لیکې_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_بنسټ;
    ثابت فراغ * src1_معلومات;
    get_لیکې_read_شاخص_fn_t    read_شاخص;
    get_لیکې_flat_compute_fn_t  compute_flat;
    get_لیکې_tiled_compute_fn_t compute_tiled;
    uint32_t task_پیل;
    uint32_t tasks;
    uint32_t tasks_فی_thread;
    uint32_t tile_اندازه;
    uint32_t tile_stride;
    uint32_t src0_لیکه_اندازه;
    uint32_t dst_لیکه_اندازه;
    صحيح idx_type;
};

#تعریف_کړه get_لیکې_preamble                      \
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

ساكن uint32_t get_لیکې_read_شاخص_ddr(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    أرجع *(ثابت uint32_t *) ((ثابت uint8_t *) src1 + off);
}

ساكن __attribute__((noinline)) uint32_t get_لیکې_read_شاخص_vtcm(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    ثابت HVX_ویکټور v = hvx_vmem((ثابت uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    أرجع (uint32_t) Q6_R_vextract_VR(v, off);
}

ساكن فراغ get_لیکې_thread_sametype(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية get_لیکې_context * grctx = (بنية get_لیکې_context *) معلومات;
    بنية htp_ops_context * octx = grctx->octx;
    ثابت بنية htp_get_لیکې_kernel_params * kparams = grctx->kparams;
    get_لیکې_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_پیل + dr * ith;
    إذا (ir0 >= grctx->task_پیل + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_پیل + grctx->tasks);
    ثابت uint32_t لیکه_اندازه_بایټونه = grctx->src0_لیکه_اندازه;
    dma_کتار * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t i = ir0; i < ir1; ++i) {
        ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ثابت uint32_t rem = i - i12 * ne11 * ne10;
        ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ثابت uint32_t i10 = rem - i11 * ne10;

        ثابت uint32_t i01 = grctx->read_شاخص(grctx->src1_معلومات, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        ثابت dma_addr_t src0_معلومات = octx->src[0]->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
        ثابت dma_addr_t dst_معلومات  = octx->dst->معلومات    + i10*nb1  + i11*nb2  + i12*nb3;

        طالما (!dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_معلومات, src0_معلومات), nb1, nb01, لیکه_اندازه_بایټونه, 1)) {
            dma_کتار_راوباسه(dma_q);
        }
    }
    dma_کتار_flush(dma_q);
}



بنية get_لیکې_flat_task {
    dma_addr_t src0_معلومات;
    dma_addr_t dst_معلومات;
};

ساكن مضمن بنية get_لیکې_flat_task get_لیکې_flat_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_لیکې_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ثابت بنية htp_get_لیکې_kernel_params * kparams = grctx->kparams;
    get_لیکې_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;

    بنية get_لیکې_flat_task task;
    task.src0_معلومات = 0;
    task.dst_معلومات  = octx->dst->معلومات + i10*nb1 + i11*nb2 + i12*nb3;

    إذا (need_src0) {
        ثابت uint32_t i01 = grctx->read_شاخص(grctx->src1_معلومات, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        task.src0_معلومات = octx->src[0]->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
    }

    أرجع task;
}

ساكن فراغ get_لیکې_thread_flat(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية get_لیکې_context * grctx = (بنية get_لیکې_context *) معلومات;
    بنية htp_ops_context * octx = grctx->octx;
    get_لیکې_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_پیل + dr * ith;
    إذا (ir0 >= grctx->task_پیل + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_پیل + grctx->tasks);

    dma_کتار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت بنية htp_get_لیکې_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنسټ, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بایټونه_فی_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنسټ, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بایټونه_فی_thread;
    ثابت uint32_t src0_بایټونه = grctx->src0_لیکه_اندازه;
    ثابت uint32_t dst_بایټونه  = grctx->dst_لیکه_اندازه;
    get_لیکې_flat_compute_fn_t compute = grctx->compute_flat;

    لكل (uint32_t ګام = 0, spad_idx = 0; ګام < ir1 - ir0 && spad_idx < 2; ++ګام, spad_idx++) {
        ثابت uint32_t i = ir0 + ګام;
        بنية get_لیکې_flat_task task = get_لیکې_flat_calc_task(octx, grctx, i, true);

        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(octx->dst->معلومات,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_اندازه),
                       dst_بایټونه, vtcm_layout->dst_spad_half_اندازه, dst_بایټونه, 0);
        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_اندازه, task.src0_معلومات),
                       vtcm_layout->src0_spad_half_اندازه, src0_بایټونه, src0_بایټونه, 1);
    }

    لكل (uint32_t ګام = 0; ګام < ir1 - ir0; ++ګام) {
        ثابت uint32_t i = ir0 + ګام;
        فراغ * dst_spad = (فراغ *) dma_کتار_راوباسه(dma_q).src;
        فراغ * src_spad = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

        بنية get_لیکې_flat_task task = get_لیکې_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((عائم *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(task.dst_معلومات, dst_spad),
                       dst_بایټونه, vtcm_layout->dst_spad_half_اندازه, dst_بایټونه, 1);

        ثابت uint32_t بل_ګام = ګام + 2;
        إذا (بل_ګام < ir1 - ir0) {
            ثابت uint32_t pi = ir0 + بل_ګام;
            بنية get_لیکې_flat_task بل_task = get_لیکې_flat_calc_task(octx, grctx, pi, true);
            dma_کتار_ټېل(dma_q,
                           dma_make_معلومات(src_spad, بل_task.src0_معلومات),
                           vtcm_layout->src0_spad_half_اندازه, src0_بایټونه, src0_بایټونه, 1);
        }
    }

    dma_کتار_flush(dma_q);
}



بنية get_لیکې_tiled_task {
    dma_addr_t tile_src_بنسټ;
    dma_addr_t dst_معلومات;
    uint32_t   لیکه;
};

ساكن مضمن بنية get_لیکې_tiled_task get_لیکې_tiled_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_لیکې_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_اندازه
) {
    ثابت بنية htp_get_لیکې_kernel_params * kparams = grctx->kparams;
    get_لیکې_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;
    ثابت uint32_t i01 = grctx->read_شاخص(grctx->src1_معلومات, i, grctx->idx_type);
    أكد(i01 < ne01);

    ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ثابت uint32_t i02 = i11 - q02 * ne02;
    ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ثابت uint32_t i03 = i12 - q03 * ne03;
    ثابت uint32_t ستنه_tile = i01 / HTP_TENSOR_TILE_N_لیکې;
    ثابت uint32_t لیکه = i01 % HTP_TENSOR_TILE_N_لیکې;
    ثابت dma_addr_t ماتریکس = octx->src[0]->معلومات + i02*nb02 + i03*nb03;

    بنية get_لیکې_tiled_task task;
    task.tile_src_بنسټ = ماتریکس + (ستنه_tile * n_k_tiles) * tile_اندازه;
    task.dst_معلومات      = octx->dst->معلومات + i10*nb1 + i11*nb2 + i12*nb3;
    task.لیکه           = لیکه;
    أرجع task;
}

ساكن فراغ get_لیکې_thread_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية get_لیکې_context * grctx = (بنية get_لیکې_context *) معلومات;
    بنية htp_ops_context * octx = grctx->octx;
    get_لیکې_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_پیل + dr * ith;
    إذا (ir0 >= grctx->task_پیل + grctx->tasks) {
        أرجع;
    }

    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_پیل + grctx->tasks);
    ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ثابت بنية htp_get_لیکې_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنسټ, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بایټونه_فی_thread;
    uint8_t * dst_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_بنسټ, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بایټونه_فی_thread;
    dma_کتار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t tile_اندازه   = grctx->tile_اندازه;
    ثابت uint32_t tile_stride = grctx->tile_stride;
    ثابت uint32_t dst_بایټونه   = grctx->dst_لیکه_اندازه;
    get_لیکې_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    لكل (uint32_t ګام = 0, spad_idx = 0; ګام < ir1 - ir0 && spad_idx < 2; ++ګام, ++spad_idx) {
        ثابت uint32_t i = ir0 + ګام;
        بنية get_لیکې_tiled_task task = get_لیکې_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_اندازه);

        // Dummy writeback to prime the queue with dst descriptor
        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(task.dst_معلومات, dst_spad_بنسټ + spad_idx * vtcm_layout->dst_spad_half_اندازه),
                       dst_بایټونه, vtcm_layout->dst_spad_half_اندازه, dst_بایټونه, 0);

        // Prefetch row tiles
        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(src_spad_بنسټ + spad_idx * vtcm_layout->src0_spad_half_اندازه, task.tile_src_بنسټ),
                       tile_stride, tile_اندازه, tile_اندازه, n_k_tiles);
    }

    لكل (uint32_t ګام = 0; ګام < ir1 - ir0; ++ګام) {
        ثابت uint32_t i = ir0 + ګام;
        عائم * dst_spad   = (عائم *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        بنية get_لیکې_tiled_task task = get_لیکې_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_اندازه);

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.لیکه);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(task.dst_معلومات, dst_spad),
                       dst_بایټونه, vtcm_layout->dst_spad_half_اندازه, dst_بایټونه, 1);

        ثابت uint32_t بل_ګام = ګام + 2;
        إذا (بل_ګام < ir1 - ir0) {
            ثابت uint32_t ni = ir0 + بل_ګام;
            بنية get_لیکې_tiled_task بل_task = get_لیکې_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_اندازه);
            dma_کتار_ټېل(dma_q,
                           dma_make_معلومات(src_spad, بل_task.tile_src_بنسټ),
                           tile_stride, tile_اندازه, tile_اندازه, n_k_tiles);
        }
    }

    dma_کتار_flush(dma_q);
}

ساكن __attribute__((noinline)) فراغ get_لیکې_dma_src1(
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

    dma_کتار * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    لكل (uint32_t i13 = 0; i13 < ne13; ++i13) {
        لكل (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ثابت dma_addr_t src_plane = octx->src[1]->معلومات + i12 * nb12 + i13 * nb13;
            طالما (!dma_کتار_ټېل(dma_q, dma_make_معلومات(cur_dst, src_plane),
                                   ne10 * elem_اندازه, nb11, ne10 * elem_اندازه, ne11)) {
                dma_کتار_راوباسه(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_اندازه;
        }
    }
    dma_کتار_flush(dma_q);
}

صحيح op_get_لیکې(بنية htp_ops_context * octx) {
    ثابت بنية htp_get_لیکې_kernel_params * kparams = (ثابت بنية htp_get_لیکې_kernel_params *) octx->kernel_params;

    ثابت uint32_t ne00          = octx->src[0]->ne[0];
    ثابت uint32_t src0_لیکه_اندازه = htp_tensor_get_لیکه_اندازه(octx->src[0]->type, ne00);
    إذا (src0_لیکه_اندازه == 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (kparams->kernel_type == HTP_GET_لیکې_KERNEL_SAMETYPE) {
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
    ثابت uint32_t ټول_tasks    = kparams->ټول_tasks;
    ثابت uint32_t dst_لیکه_اندازه   = (kparams->kernel_type == HTP_GET_لیکې_KERNEL_SAMETYPE) ? src0_لیکه_اندازه : ne00 * حجم(عائم);

    uint32_t task_پیل = 0;
    uint32_t tasks      = ټول_tasks;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t tasks_فی_chunk = 1;
        htp_tensor_mdev_لیکې_فی_chunk(dst, dst_لیکه_اندازه / dst->ne[0], dst_لیکه_اندازه, &tasks_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ټول_tasks, tasks_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        task_پیل = range.پیل;
        tasks      = range.شمېر;
    }

    إذا (tasks == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, (uint32_t) kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    بنية get_لیکې_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_بنسټ = (uint8_t *)octx->ctx->vtcm_بنسټ;
    grctx.compute_flat = عدم;
    grctx.compute_tiled = عدم;
    grctx.task_پیل = task_پیل;
    grctx.tasks = tasks;
    grctx.tasks_فی_thread = octx->ctx->mdev.شمېر == 1 ? kparams->tasks_فی_thread : fastdiv(tasks + n_تارونه - 1, &octx->n_تارونه_div);
    grctx.tile_اندازه   = 0;
    grctx.tile_stride = 0;
    grctx.src0_لیکه_اندازه = src0_لیکه_اندازه;
    grctx.dst_لیکه_اندازه  = dst_لیکه_اندازه;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_لیکې_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, ټول_tasks, grctx.idx_type, n_تارونه);

    إذا (grctx.vtcm_layout.ټول_بایټونه > octx->ctx->vtcm_اندازه) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_اندازه, grctx.vtcm_layout.ټول_بایټونه);
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    work_کتار_func_t q_func = عدم;
    بدل (kparams->kernel_type) {
        حالة HTP_GET_لیکې_KERNEL_SAMETYPE:
            q_func = get_لیکې_thread_sametype;
            اكسر;
        حالة HTP_GET_لیکې_KERNEL_TILED:
            بدل (octx->src[0]->type) {
                حالة HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_لیکه_tiled_f16_f32;  اكسر;
                حالة HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_لیکه_tiled_q1_0_f32; اكسر;
                حالة HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_لیکه_tiled_q2_0_f32; اكسر;
                حالة HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_لیکه_tiled_q8_0_f32; اكسر;
                حالة HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_لیکه_tiled_q4_0_f32; اكسر;
                حالة HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_لیکه_tiled_q4_k_f32; اكسر;
                حالة HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_لیکه_tiled_q5_k_f32; اكسر;
                حالة HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_لیکه_tiled_q6_k_f32; اكسر;
                افتراضي:            أرجع HTP_حالت_NO_SUPPORT;
            }
            grctx.tile_اندازه   = htp_tensor_get_tile_اندازه(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_اندازه(octx->src[0]->type);
            q_func = get_لیکې_thread_tiled;
            اكسر;
        حالة HTP_GET_لیکې_KERNEL_FLAT:
            بدل (octx->src[0]->type) {
                حالة HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_لیکه_flat_f16_f32;  اكسر;
                حالة HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_لیکه_flat_bf16_f32; اكسر;
                حالة HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_لیکه_flat_q1_0_f32; اكسر;
                حالة HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_لیکه_flat_q2_0_f32; اكسر;
                حالة HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_لیکه_flat_q8_0_f32; اكسر;
                حالة HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_لیکه_flat_q4_0_f32; اكسر;
                حالة HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_لیکه_flat_q4_k_f32; اكسر;
                حالة HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_لیکه_flat_q5_k_f32; اكسر;
                حالة HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_لیکه_flat_q6_k_f32; اكسر;
                افتراضي:            أرجع HTP_حالت_NO_SUPPORT;
            }
            q_func = get_لیکې_thread_flat;
            اكسر;
        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t elem_اندازه = (grctx.idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_بایټونه  = (size_t) ټول_tasks * elem_اندازه;

    إذا (!htp_tensor_is_extended(octx->src[1]) && src1_بایټونه <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_اندازه)) {
        grctx.src1_معلومات  = (ثابت فراغ *) (uintptr_t) octx->src[1]->معلومات;
        grctx.read_شاخص = get_لیکې_read_شاخص_ddr;
    } وإلا {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_بنسټ, grctx.vtcm_layout.off_src1);
        get_لیکې_dma_src1(octx, vtcm_src1, elem_اندازه);
        grctx.src1_معلومات  = vtcm_src1;
        grctx.read_شاخص = get_لیکې_read_شاخص_vtcm;
    }

    FARF(لوړ_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_بایټونه_فی_thread * n_تارونه,
         grctx.vtcm_layout.dst_بایټونه_فی_thread  * n_تارونه,
         kparams->kernel_type, n_تارونه);

    work_کتار_پرلپسې(octx->ctx->work_کتار, q_func, &grctx, n_تارونه);
    أرجع HTP_حالت_OK;
}
