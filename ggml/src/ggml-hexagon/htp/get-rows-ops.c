#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#تعریف_کرو GGML_COMMON_DECL_C
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

بنية get_صفاں_context;

تعريفنوع uint32_t (*get_صفاں_read_اشاریہ_fn_t)(ثابت فراغ * src1, uint32_t idx, صحيح idx_type);
تعريفنوع فراغ (*get_صفاں_flat_compute_fn_t)(عائم * dst_spad, ثابت فراغ * src_spad, uint32_t cur_elems);
تعريفنوع فراغ (*get_صفاں_tiled_compute_fn_t)(عائم * dst_spad, ثابت uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t صف);

بنية get_صفاں_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_get_صفاں_kernel_params * kparams;
    بنية htp_get_صفاں_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_ادھار;
    ثابت فراغ * src1_انکڑے;
    get_صفاں_read_اشاریہ_fn_t    read_اشاریہ;
    get_صفاں_flat_compute_fn_t  compute_flat;
    get_صفاں_tiled_compute_fn_t compute_tiled;
    uint32_t task_شروع;
    uint32_t tasks;
    uint32_t tasks_فی_thread;
    uint32_t tile_ناپ;
    uint32_t tile_stride;
    uint32_t src0_صف_ناپ;
    uint32_t dst_صف_ناپ;
    صحيح idx_type;
};

#تعریف_کرو get_صفاں_preamble                      \
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

ساكن uint32_t get_صفاں_read_اشاریہ_ddr(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    أرجع *(ثابت uint32_t *) ((ثابت uint8_t *) src1 + off);
}

ساكن __attribute__((noinline)) uint32_t get_صفاں_read_اشاریہ_vtcm(
    ثابت فراغ * src1,
    uint32_t idx,
    صحيح idx_type
) {
    ثابت uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t));
    ثابت HVX_سمتیہ v = hvx_vmem((ثابت uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    أرجع (uint32_t) Q6_R_vextract_VR(v, off);
}

ساكن فراغ get_صفاں_thread_sametype(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية get_صفاں_context * grctx = (بنية get_صفاں_context *) انکڑے;
    بنية htp_ops_context * octx = grctx->octx;
    ثابت بنية htp_get_صفاں_kernel_params * kparams = grctx->kparams;
    get_صفاں_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_شروع + dr * ith;
    إذا (ir0 >= grctx->task_شروع + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروع + grctx->tasks);
    ثابت uint32_t صف_ناپ_بائٹاں = grctx->src0_صف_ناپ;
    dma_قطار * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t i = ir0; i < ir1; ++i) {
        ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ثابت uint32_t rem = i - i12 * ne11 * ne10;
        ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ثابت uint32_t i10 = rem - i11 * ne10;

        ثابت uint32_t i01 = grctx->read_اشاریہ(grctx->src1_انکڑے, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        ثابت dma_addr_t src0_انکڑے = octx->src[0]->انکڑے + i01*nb01 + i02*nb02 + i03*nb03;
        ثابت dma_addr_t dst_انکڑے  = octx->dst->انکڑے    + i10*nb1  + i11*nb2  + i12*nb3;

        طالما (!dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst_انکڑے, src0_انکڑے), nb1, nb01, صف_ناپ_بائٹاں, 1)) {
            dma_قطار_نکالو(dma_q);
        }
    }
    dma_قطار_flush(dma_q);
}



بنية get_صفاں_flat_task {
    dma_addr_t src0_انکڑے;
    dma_addr_t dst_انکڑے;
};

ساكن مضمن بنية get_صفاں_flat_task get_صفاں_flat_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_صفاں_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ثابت بنية htp_get_صفاں_kernel_params * kparams = grctx->kparams;
    get_صفاں_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;

    بنية get_صفاں_flat_task task;
    task.src0_انکڑے = 0;
    task.dst_انکڑے  = octx->dst->انکڑے + i10*nb1 + i11*nb2 + i12*nb3;

    إذا (need_src0) {
        ثابت uint32_t i01 = grctx->read_اشاریہ(grctx->src1_انکڑے, i, grctx->idx_type);
        أكد(i01 < ne01);

        ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ثابت uint32_t i02 = i11 - q02 * ne02;
        ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ثابت uint32_t i03 = i12 - q03 * ne03;

        task.src0_انکڑے = octx->src[0]->انکڑے + i01*nb01 + i02*nb02 + i03*nb03;
    }

    أرجع task;
}

ساكن فراغ get_صفاں_thread_flat(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية get_صفاں_context * grctx = (بنية get_صفاں_context *) انکڑے;
    بنية htp_ops_context * octx = grctx->octx;
    get_صفاں_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_شروع + dr * ith;
    إذا (ir0 >= grctx->task_شروع + grctx->tasks) {
        أرجع;
    }
    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروع + grctx->tasks);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت بنية htp_get_صفاں_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ادھار, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بائٹاں_فی_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ادھار, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بائٹاں_فی_thread;
    ثابت uint32_t src0_بائٹاں = grctx->src0_صف_ناپ;
    ثابت uint32_t dst_بائٹاں  = grctx->dst_صف_ناپ;
    get_صفاں_flat_compute_fn_t compute = grctx->compute_flat;

    لكل (uint32_t قدم = 0, spad_idx = 0; قدم < ir1 - ir0 && spad_idx < 2; ++قدم, spad_idx++) {
        ثابت uint32_t i = ir0 + قدم;
        بنية get_صفاں_flat_task task = get_صفاں_flat_calc_task(octx, grctx, i, true);

        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(octx->dst->انکڑے,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_ناپ),
                       dst_بائٹاں, vtcm_layout->dst_spad_half_ناپ, dst_بائٹاں, 0);
        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_ناپ, task.src0_انکڑے),
                       vtcm_layout->src0_spad_half_ناپ, src0_بائٹاں, src0_بائٹاں, 1);
    }

    لكل (uint32_t قدم = 0; قدم < ir1 - ir0; ++قدم) {
        ثابت uint32_t i = ir0 + قدم;
        فراغ * dst_spad = (فراغ *) dma_قطار_نکالو(dma_q).src;
        فراغ * src_spad = (فراغ *) dma_قطار_نکالو(dma_q).dst;

        بنية get_صفاں_flat_task task = get_صفاں_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((عائم *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(task.dst_انکڑے, dst_spad),
                       dst_بائٹاں, vtcm_layout->dst_spad_half_ناپ, dst_بائٹاں, 1);

        ثابت uint32_t اگلا_قدم = قدم + 2;
        إذا (اگلا_قدم < ir1 - ir0) {
            ثابت uint32_t pi = ir0 + اگلا_قدم;
            بنية get_صفاں_flat_task اگلا_task = get_صفاں_flat_calc_task(octx, grctx, pi, true);
            dma_قطار_ڈالو(dma_q,
                           dma_make_انکڑے(src_spad, اگلا_task.src0_انکڑے),
                           vtcm_layout->src0_spad_half_ناپ, src0_بائٹاں, src0_بائٹاں, 1);
        }
    }

    dma_قطار_flush(dma_q);
}



بنية get_صفاں_tiled_task {
    dma_addr_t tile_src_ادھار;
    dma_addr_t dst_انکڑے;
    uint32_t   صف;
};

ساكن مضمن بنية get_صفاں_tiled_task get_صفاں_tiled_calc_task(
    ثابت بنية htp_ops_context * octx,
    ثابت بنية get_صفاں_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_ناپ
) {
    ثابت بنية htp_get_صفاں_kernel_params * kparams = grctx->kparams;
    get_صفاں_preamble;

    ثابت uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ثابت uint32_t rem = i - i12 * ne11 * ne10;
    ثابت uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ثابت uint32_t i10 = rem - i11 * ne10;
    ثابت uint32_t i01 = grctx->read_اشاریہ(grctx->src1_انکڑے, i, grctx->idx_type);
    أكد(i01 < ne01);

    ثابت uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ثابت uint32_t i02 = i11 - q02 * ne02;
    ثابت uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ثابت uint32_t i03 = i12 - q03 * ne03;
    ثابت uint32_t کالم_tile = i01 / HTP_TENSOR_TILE_N_صفاں;
    ثابت uint32_t صف = i01 % HTP_TENSOR_TILE_N_صفاں;
    ثابت dma_addr_t میٹرکس = octx->src[0]->انکڑے + i02*nb02 + i03*nb03;

    بنية get_صفاں_tiled_task task;
    task.tile_src_ادھار = میٹرکس + (کالم_tile * n_k_tiles) * tile_ناپ;
    task.dst_انکڑے      = octx->dst->انکڑے + i10*nb1 + i11*nb2 + i12*nb3;
    task.صف           = صف;
    أرجع task;
}

ساكن فراغ get_صفاں_thread_tiled(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية get_صفاں_context * grctx = (بنية get_صفاں_context *) انکڑے;
    بنية htp_ops_context * octx = grctx->octx;
    get_صفاں_preamble;

    ثابت uint32_t dr  = grctx->tasks_فی_thread;
    ثابت uint32_t ir0 = grctx->task_شروع + dr * ith;
    إذا (ir0 >= grctx->task_شروع + grctx->tasks) {
        أرجع;
    }

    ثابت uint32_t ir1 = MIN(ir0 + dr, grctx->task_شروع + grctx->tasks);
    ثابت uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ثابت بنية htp_get_صفاں_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_ادھار = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ادھار, vtcm_layout->off_src0) + ith * vtcm_layout->src0_بائٹاں_فی_thread;
    uint8_t * dst_spad_ادھار = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ادھار, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_بائٹاں_فی_thread;
    dma_قطار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t tile_ناپ   = grctx->tile_ناپ;
    ثابت uint32_t tile_stride = grctx->tile_stride;
    ثابت uint32_t dst_بائٹاں   = grctx->dst_صف_ناپ;
    get_صفاں_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    لكل (uint32_t قدم = 0, spad_idx = 0; قدم < ir1 - ir0 && spad_idx < 2; ++قدم, ++spad_idx) {
        ثابت uint32_t i = ir0 + قدم;
        بنية get_صفاں_tiled_task task = get_صفاں_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_ناپ);

        // Dummy writeback to prime the queue with dst descriptor
        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(task.dst_انکڑے, dst_spad_ادھار + spad_idx * vtcm_layout->dst_spad_half_ناپ),
                       dst_بائٹاں, vtcm_layout->dst_spad_half_ناپ, dst_بائٹاں, 0);

        // Prefetch row tiles
        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(src_spad_ادھار + spad_idx * vtcm_layout->src0_spad_half_ناپ, task.tile_src_ادھار),
                       tile_stride, tile_ناپ, tile_ناپ, n_k_tiles);
    }

    لكل (uint32_t قدم = 0; قدم < ir1 - ir0; ++قدم) {
        ثابت uint32_t i = ir0 + قدم;
        عائم * dst_spad   = (عائم *) dma_قطار_نکالو(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        بنية get_صفاں_tiled_task task = get_صفاں_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_ناپ);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.صف);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(task.dst_انکڑے, dst_spad),
                       dst_بائٹاں, vtcm_layout->dst_spad_half_ناپ, dst_بائٹاں, 1);

        ثابت uint32_t اگلا_قدم = قدم + 2;
        إذا (اگلا_قدم < ir1 - ir0) {
            ثابت uint32_t ni = ir0 + اگلا_قدم;
            بنية get_صفاں_tiled_task اگلا_task = get_صفاں_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_ناپ);
            dma_قطار_ڈالو(dma_q,
                           dma_make_انکڑے(src_spad, اگلا_task.tile_src_ادھار),
                           tile_stride, tile_ناپ, tile_ناپ, n_k_tiles);
        }
    }

    dma_قطار_flush(dma_q);
}

ساكن __attribute__((noinline)) فراغ get_صفاں_dma_src1(
    بنية htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_ناپ
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
            ثابت dma_addr_t src_plane = octx->src[1]->انکڑے + i12 * nb12 + i13 * nb13;
            طالما (!dma_قطار_ڈالو(dma_q, dma_make_انکڑے(cur_dst, src_plane),
                                   ne10 * elem_ناپ, nb11, ne10 * elem_ناپ, ne11)) {
                dma_قطار_نکالو(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_ناپ;
        }
    }
    dma_قطار_flush(dma_q);
}

صحيح op_get_صفاں(بنية htp_ops_context * octx) {
    ثابت بنية htp_get_صفاں_kernel_params * kparams = (ثابت بنية htp_get_صفاں_kernel_params *) octx->kernel_params;

    ثابت uint32_t ne00          = octx->src[0]->ne[0];
    ثابت uint32_t src0_صف_ناپ = htp_tensor_get_صف_ناپ(octx->src[0]->type, ne00);
    إذا (src0_صف_ناپ == 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (kparams->kernel_type == HTP_GET_صفاں_KERNEL_SAMETYPE) {
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
    ثابت uint32_t کل_tasks    = kparams->کل_tasks;
    ثابت uint32_t dst_صف_ناپ   = (kparams->kernel_type == HTP_GET_صفاں_KERNEL_SAMETYPE) ? src0_صف_ناپ : ne00 * حجم(عائم);

    uint32_t task_شروع = 0;
    uint32_t tasks      = کل_tasks;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t tasks_فی_chunk = 1;
        htp_tensor_mdev_صفاں_فی_chunk(dst, dst_صف_ناپ / dst->ne[0], dst_صف_ناپ, &tasks_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(کل_tasks, tasks_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        task_شروع = range.شروع;
        tasks      = range.گنتی;
    }

    إذا (tasks == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (!htp_ops_context_سیٹ_n_دھاگے(octx, (uint32_t) kparams->n_دھاگے)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    بنية get_صفاں_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_ادھار = (uint8_t *)octx->ctx->vtcm_ادھار;
    grctx.compute_flat = عدم;
    grctx.compute_tiled = عدم;
    grctx.task_شروع = task_شروع;
    grctx.tasks = tasks;
    grctx.tasks_فی_thread = octx->ctx->mdev.گنتی == 1 ? kparams->tasks_فی_thread : fastdiv(tasks + n_دھاگے - 1, &octx->n_دھاگے_div);
    grctx.tile_ناپ   = 0;
    grctx.tile_stride = 0;
    grctx.src0_صف_ناپ = src0_صف_ناپ;
    grctx.dst_صف_ناپ  = dst_صف_ناپ;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_صفاں_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, کل_tasks, grctx.idx_type, n_دھاگے);

    إذا (grctx.vtcm_layout.کل_بائٹاں > octx->ctx->vtcm_ناپ) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ناپ, grctx.vtcm_layout.کل_بائٹاں);
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    work_قطار_func_t q_func = عدم;
    بدل (kparams->kernel_type) {
        حالة HTP_GET_صفاں_KERNEL_SAMETYPE:
            q_func = get_صفاں_thread_sametype;
            اكسر;
        حالة HTP_GET_صفاں_KERNEL_TILED:
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
            grctx.tile_ناپ   = htp_tensor_get_tile_ناپ(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_ناپ(octx->src[0]->type);
            q_func = get_صفاں_thread_tiled;
            اكسر;
        حالة HTP_GET_صفاں_KERNEL_FLAT:
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
            q_func = get_صفاں_thread_flat;
            اكسر;
        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t elem_ناپ = (grctx.idx_type == HTP_TYPE_I64) ? حجم(int64_t) : حجم(int32_t);
    ثابت size_t src1_بائٹاں  = (size_t) کل_tasks * elem_ناپ;

    إذا (!htp_tensor_is_extended(octx->src[1]) && src1_بائٹاں <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_ناپ)) {
        grctx.src1_انکڑے  = (ثابت فراغ *) (uintptr_t) octx->src[1]->انکڑے;
        grctx.read_اشاریہ = get_صفاں_read_اشاریہ_ddr;
    } وإلا {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_ادھار, grctx.vtcm_layout.off_src1);
        get_صفاں_dma_src1(octx, vtcm_src1, elem_ناپ);
        grctx.src1_انکڑے  = vtcm_src1;
        grctx.read_اشاریہ = get_صفاں_read_اشاریہ_vtcm;
    }

    FARF(بلند_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_بائٹاں_فی_thread * n_دھاگے,
         grctx.vtcm_layout.dst_بائٹاں_فی_thread  * n_دھاگے,
         kparams->kernel_type, n_دھاگے);

    work_قطار_لگاتار(octx->ctx->work_قطار, q_func, &grctx, n_دھاگے);
    أرجع HTP_حالت_OK;
}
