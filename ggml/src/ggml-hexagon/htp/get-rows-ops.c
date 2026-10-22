#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-quant.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "get-rows-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"

ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint32_t (*get_ਪੰਕਤੀਆਂ_read_ਸੂਚਕਾਂਕ_fn_t)(ਅਚਲ ਖਾਲੀ * src1, uint32_t idx, ਪੂਰਨ idx_type);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*get_ਪੰਕਤੀਆਂ_flat_compute_fn_t)(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ ਖਾਲੀ * src_spad, uint32_t cur_elems);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*get_ਪੰਕਤੀਆਂ_tiled_compute_fn_t)(ਦਸ਼ਮਲਵ * dst_spad, ਅਚਲ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ਪੰਕਤੀ);

ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params * kparams;
    ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_ਅਧਾਰ;
    ਅਚਲ ਖਾਲੀ * src1_ਅੰਕੜੇ;
    get_ਪੰਕਤੀਆਂ_read_ਸੂਚਕਾਂਕ_fn_t    read_ਸੂਚਕਾਂਕ;
    get_ਪੰਕਤੀਆਂ_flat_compute_fn_t  compute_flat;
    get_ਪੰਕਤੀਆਂ_tiled_compute_fn_t compute_tiled;
    uint32_t task_ਸ਼ੁਰੂ;
    uint32_t tasks;
    uint32_t tasks_ਪ੍ਰਤੀ_thread;
    uint32_t tile_ਮਾਪ;
    uint32_t tile_stride;
    uint32_t src0_ਪੰਕਤੀ_ਮਾਪ;
    uint32_t dst_ਪੰਕਤੀ_ਮਾਪ;
    ਪੂਰਨ idx_type;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ get_ਪੰਕਤੀਆਂ_preamble                      \
    ਅਚਲ uint32_t ne00 = octx->src[0]->ne[0]; \
    ਅਚਲ uint32_t ne01 = octx->src[0]->ne[1]; \
    ਅਚਲ uint32_t ne02 = octx->src[0]->ne[2]; \
    ਅਚਲ uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ਅਚਲ uint32_t ne10 = octx->src[1]->ne[0]; \
    ਅਚਲ uint32_t ne11 = octx->src[1]->ne[1]; \
    ਅਚਲ uint32_t ne12 = octx->src[1]->ne[2]; \
    ਅਚਲ uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ਅਚਲ uint32_t ne0 = octx->dst->ne[0];     \
    ਅਚਲ uint32_t ne1 = octx->dst->ne[1];     \
    ਅਚਲ uint32_t ne2 = octx->dst->ne[2];     \
    ਅਚਲ uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ਅਚਲ uint32_t nb01 = octx->src[0]->nb[1]; \
    ਅਚਲ uint32_t nb02 = octx->src[0]->nb[2]; \
    ਅਚਲ uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ਅਚਲ uint32_t nb10 = octx->src[1]->nb[0]; \
    ਅਚਲ uint32_t nb11 = octx->src[1]->nb[1]; \
    ਅਚਲ uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ਅਚਲ uint32_t nb1 = octx->dst->nb[1];     \
    ਅਚਲ uint32_t nb2 = octx->dst->nb[2];     \
    ਅਚਲ uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ਅਚਲ uint32_t nr = ne10 * ne11 * ne12;

ਸਥਿਰ uint32_t get_ਪੰਕਤੀਆਂ_read_ਸੂਚਕਾਂਕ_ddr(
    ਅਚਲ ਖਾਲੀ * src1,
    uint32_t idx,
    ਪੂਰਨ idx_type
) {
    ਅਚਲ uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? ਆਕਾਰ(int64_t) : ਆਕਾਰ(int32_t));
    ਵਾਪਸ *(ਅਚਲ uint32_t *) ((ਅਚਲ uint8_t *) src1 + off);
}

ਸਥਿਰ __attribute__((noinline)) uint32_t get_ਪੰਕਤੀਆਂ_read_ਸੂਚਕਾਂਕ_vtcm(
    ਅਚਲ ਖਾਲੀ * src1,
    uint32_t idx,
    ਪੂਰਨ idx_type
) {
    ਅਚਲ uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? ਆਕਾਰ(int64_t) : ਆਕਾਰ(int32_t));
    ਅਚਲ HVX_ਸਦਿਸ਼ v = hvx_vmem((ਅਚਲ uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    ਵਾਪਸ (uint32_t) Q6_R_vextract_VR(v, off);
}

ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_thread_sametype(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context * grctx = (ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = grctx->octx;
    ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params * kparams = grctx->kparams;
    get_ਪੰਕਤੀਆਂ_preamble;

    ਅਚਲ uint32_t dr  = grctx->tasks_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ir0 = grctx->task_ਸ਼ੁਰੂ + dr * ith;
    ਜੇ (ir0 >= grctx->task_ਸ਼ੁਰੂ + grctx->tasks) {
        ਵਾਪਸ;
    }
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, grctx->task_ਸ਼ੁਰੂ + grctx->tasks);
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = grctx->src0_ਪੰਕਤੀ_ਮਾਪ;
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਲਈ (uint32_t i = ir0; i < ir1; ++i) {
        ਅਚਲ uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ਅਚਲ uint32_t rem = i - i12 * ne11 * ne10;
        ਅਚਲ uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ਅਚਲ uint32_t i10 = rem - i11 * ne10;

        ਅਚਲ uint32_t i01 = grctx->read_ਸੂਚਕਾਂਕ(grctx->src1_ਅੰਕੜੇ, i, grctx->idx_type);
        ਪ੍ਰਤਿਜ੍ਞਾ(i01 < ne01);

        ਅਚਲ uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ਅਚਲ uint32_t i02 = i11 - q02 * ne02;
        ਅਚਲ uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ਅਚਲ uint32_t i03 = i12 - q03 * ne03;

        ਅਚਲ dma_addr_t src0_ਅੰਕੜੇ = octx->src[0]->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;
        ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ  = octx->dst->ਅੰਕੜੇ    + i10*nb1  + i11*nb2  + i12*nb3;

        ਜਦੋਂ (!dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, src0_ਅੰਕੜੇ), nb1, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, 1)) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }
    }
    dma_ਕਤਾਰ_flush(dma_q);
}



ਬਣਤਰ get_ਪੰਕਤੀਆਂ_flat_task {
    dma_addr_t src0_ਅੰਕੜੇ;
    dma_addr_t dst_ਅੰਕੜੇ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ get_ਪੰਕਤੀਆਂ_flat_task get_ਪੰਕਤੀਆਂ_flat_calc_task(
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params * kparams = grctx->kparams;
    get_ਪੰਕਤੀਆਂ_preamble;

    ਅਚਲ uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ਅਚਲ uint32_t rem = i - i12 * ne11 * ne10;
    ਅਚਲ uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ਅਚਲ uint32_t i10 = rem - i11 * ne10;

    ਬਣਤਰ get_ਪੰਕਤੀਆਂ_flat_task task;
    task.src0_ਅੰਕੜੇ = 0;
    task.dst_ਅੰਕੜੇ  = octx->dst->ਅੰਕੜੇ + i10*nb1 + i11*nb2 + i12*nb3;

    ਜੇ (need_src0) {
        ਅਚਲ uint32_t i01 = grctx->read_ਸੂਚਕਾਂਕ(grctx->src1_ਅੰਕੜੇ, i, grctx->idx_type);
        ਪ੍ਰਤਿਜ੍ਞਾ(i01 < ne01);

        ਅਚਲ uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ਅਚਲ uint32_t i02 = i11 - q02 * ne02;
        ਅਚਲ uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ਅਚਲ uint32_t i03 = i12 - q03 * ne03;

        task.src0_ਅੰਕੜੇ = octx->src[0]->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;
    }

    ਵਾਪਸ task;
}

ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_thread_flat(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context * grctx = (ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = grctx->octx;
    get_ਪੰਕਤੀਆਂ_preamble;

    ਅਚਲ uint32_t dr  = grctx->tasks_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ir0 = grctx->task_ਸ਼ੁਰੂ + dr * ith;
    ਜੇ (ir0 >= grctx->task_ਸ਼ੁਰੂ + grctx->tasks) {
        ਵਾਪਸ;
    }
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, grctx->task_ਸ਼ੁਰੂ + grctx->tasks);

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ਅਧਾਰ, vtcm_layout->off_src0) + ith * vtcm_layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ਅਧਾਰ, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t src0_ਬਾਈਟਾਂ = grctx->src0_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ uint32_t dst_ਬਾਈਟਾਂ  = grctx->dst_ਪੰਕਤੀ_ਮਾਪ;
    get_ਪੰਕਤੀਆਂ_flat_compute_fn_t compute = grctx->compute_flat;

    ਲਈ (uint32_t ਕਦਮ = 0, spad_idx = 0; ਕਦਮ < ir1 - ir0 && spad_idx < 2; ++ਕਦਮ, spad_idx++) {
        ਅਚਲ uint32_t i = ir0 + ਕਦਮ;
        ਬਣਤਰ get_ਪੰਕਤੀਆਂ_flat_task task = get_ਪੰਕਤੀਆਂ_flat_calc_task(octx, grctx, i, true);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(octx->dst->ਅੰਕੜੇ,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_ਮਾਪ),
                       dst_ਬਾਈਟਾਂ, vtcm_layout->dst_spad_half_ਮਾਪ, dst_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_ਮਾਪ, task.src0_ਅੰਕੜੇ),
                       vtcm_layout->src0_spad_half_ਮਾਪ, src0_ਬਾਈਟਾਂ, src0_ਬਾਈਟਾਂ, 1);
    }

    ਲਈ (uint32_t ਕਦਮ = 0; ਕਦਮ < ir1 - ir0; ++ਕਦਮ) {
        ਅਚਲ uint32_t i = ir0 + ਕਦਮ;
        ਖਾਲੀ * dst_spad = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * src_spad = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        ਬਣਤਰ get_ਪੰਕਤੀਆਂ_flat_task task = get_ਪੰਕਤੀਆਂ_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((ਦਸ਼ਮਲਵ *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(task.dst_ਅੰਕੜੇ, dst_spad),
                       dst_ਬਾਈਟਾਂ, vtcm_layout->dst_spad_half_ਮਾਪ, dst_ਬਾਈਟਾਂ, 1);

        ਅਚਲ uint32_t ਅਗਲਾ_ਕਦਮ = ਕਦਮ + 2;
        ਜੇ (ਅਗਲਾ_ਕਦਮ < ir1 - ir0) {
            ਅਚਲ uint32_t pi = ir0 + ਅਗਲਾ_ਕਦਮ;
            ਬਣਤਰ get_ਪੰਕਤੀਆਂ_flat_task ਅਗਲਾ_task = get_ਪੰਕਤੀਆਂ_flat_calc_task(octx, grctx, pi, true);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                           dma_make_ਅੰਕੜੇ(src_spad, ਅਗਲਾ_task.src0_ਅੰਕੜੇ),
                           vtcm_layout->src0_spad_half_ਮਾਪ, src0_ਬਾਈਟਾਂ, src0_ਬਾਈਟਾਂ, 1);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
}



ਬਣਤਰ get_ਪੰਕਤੀਆਂ_tiled_task {
    dma_addr_t tile_src_ਅਧਾਰ;
    dma_addr_t dst_ਅੰਕੜੇ;
    uint32_t   ਪੰਕਤੀ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ get_ਪੰਕਤੀਆਂ_tiled_task get_ਪੰਕਤੀਆਂ_tiled_calc_task(
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_ਮਾਪ
) {
    ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params * kparams = grctx->kparams;
    get_ਪੰਕਤੀਆਂ_preamble;

    ਅਚਲ uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ਅਚਲ uint32_t rem = i - i12 * ne11 * ne10;
    ਅਚਲ uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ਅਚਲ uint32_t i10 = rem - i11 * ne10;
    ਅਚਲ uint32_t i01 = grctx->read_ਸੂਚਕਾਂਕ(grctx->src1_ਅੰਕੜੇ, i, grctx->idx_type);
    ਪ੍ਰਤਿਜ੍ਞਾ(i01 < ne01);

    ਅਚਲ uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ਅਚਲ uint32_t i02 = i11 - q02 * ne02;
    ਅਚਲ uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ਅਚਲ uint32_t i03 = i12 - q03 * ne03;
    ਅਚਲ uint32_t ਕਾਲਮ_tile = i01 / HTP_TENSOR_TILE_N_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t ਪੰਕਤੀ = i01 % HTP_TENSOR_TILE_N_ਪੰਕਤੀਆਂ;
    ਅਚਲ dma_addr_t ਮੈਟ੍ਰਿਕਸ = octx->src[0]->ਅੰਕੜੇ + i02*nb02 + i03*nb03;

    ਬਣਤਰ get_ਪੰਕਤੀਆਂ_tiled_task task;
    task.tile_src_ਅਧਾਰ = ਮੈਟ੍ਰਿਕਸ + (ਕਾਲਮ_tile * n_k_tiles) * tile_ਮਾਪ;
    task.dst_ਅੰਕੜੇ      = octx->dst->ਅੰਕੜੇ + i10*nb1 + i11*nb2 + i12*nb3;
    task.ਪੰਕਤੀ           = ਪੰਕਤੀ;
    ਵਾਪਸ task;
}

ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_thread_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context * grctx = (ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = grctx->octx;
    get_ਪੰਕਤੀਆਂ_preamble;

    ਅਚਲ uint32_t dr  = grctx->tasks_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ir0 = grctx->task_ਸ਼ੁਰੂ + dr * ith;
    ਜੇ (ir0 >= grctx->task_ਸ਼ੁਰੂ + grctx->tasks) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, grctx->task_ਸ਼ੁਰੂ + grctx->tasks);
    ਅਚਲ uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ਅਧਾਰ, vtcm_layout->off_src0) + ith * vtcm_layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint8_t * dst_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ਅਧਾਰ, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ uint32_t tile_ਮਾਪ   = grctx->tile_ਮਾਪ;
    ਅਚਲ uint32_t tile_stride = grctx->tile_stride;
    ਅਚਲ uint32_t dst_ਬਾਈਟਾਂ   = grctx->dst_ਪੰਕਤੀ_ਮਾਪ;
    get_ਪੰਕਤੀਆਂ_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    ਲਈ (uint32_t ਕਦਮ = 0, spad_idx = 0; ਕਦਮ < ir1 - ir0 && spad_idx < 2; ++ਕਦਮ, ++spad_idx) {
        ਅਚਲ uint32_t i = ir0 + ਕਦਮ;
        ਬਣਤਰ get_ਪੰਕਤੀਆਂ_tiled_task task = get_ਪੰਕਤੀਆਂ_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_ਮਾਪ);

        // Dummy writeback to prime the queue with dst descriptor
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(task.dst_ਅੰਕੜੇ, dst_spad_ਅਧਾਰ + spad_idx * vtcm_layout->dst_spad_half_ਮਾਪ),
                       dst_ਬਾਈਟਾਂ, vtcm_layout->dst_spad_half_ਮਾਪ, dst_ਬਾਈਟਾਂ, 0);

        // Prefetch row tiles
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(src_spad_ਅਧਾਰ + spad_idx * vtcm_layout->src0_spad_half_ਮਾਪ, task.tile_src_ਅਧਾਰ),
                       tile_stride, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles);
    }

    ਲਈ (uint32_t ਕਦਮ = 0; ਕਦਮ < ir1 - ir0; ++ਕਦਮ) {
        ਅਚਲ uint32_t i = ir0 + ਕਦਮ;
        ਦਸ਼ਮਲਵ * dst_spad   = (ਦਸ਼ਮਲਵ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        ਬਣਤਰ get_ਪੰਕਤੀਆਂ_tiled_task task = get_ਪੰਕਤੀਆਂ_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_ਮਾਪ);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.ਪੰਕਤੀ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(task.dst_ਅੰਕੜੇ, dst_spad),
                       dst_ਬਾਈਟਾਂ, vtcm_layout->dst_spad_half_ਮਾਪ, dst_ਬਾਈਟਾਂ, 1);

        ਅਚਲ uint32_t ਅਗਲਾ_ਕਦਮ = ਕਦਮ + 2;
        ਜੇ (ਅਗਲਾ_ਕਦਮ < ir1 - ir0) {
            ਅਚਲ uint32_t ni = ir0 + ਅਗਲਾ_ਕਦਮ;
            ਬਣਤਰ get_ਪੰਕਤੀਆਂ_tiled_task ਅਗਲਾ_task = get_ਪੰਕਤੀਆਂ_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_ਮਾਪ);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                           dma_make_ਅੰਕੜੇ(src_spad, ਅਗਲਾ_task.tile_src_ਅਧਾਰ),
                           tile_stride, tile_ਮਾਪ, tile_ਮਾਪ, n_k_tiles);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ get_ਪੰਕਤੀਆਂ_dma_src1(
    ਬਣਤਰ htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_ਮਾਪ
) {
    ਅਚਲ uint32_t ne10 = octx->src[1]->ne[0];
    ਅਚਲ uint32_t ne11 = octx->src[1]->ne[1];
    ਅਚਲ uint32_t ne12 = octx->src[1]->ne[2];
    ਅਚਲ uint32_t ne13 = octx->src[1]->ne[3];
    ਅਚਲ uint32_t nb11 = octx->src[1]->nb[1];
    ਅਚਲ uint32_t nb12 = octx->src[1]->nb[2];
    ਅਚਲ uint32_t nb13 = octx->src[1]->nb[3];

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    ਲਈ (uint32_t i13 = 0; i13 < ne13; ++i13) {
        ਲਈ (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ਅਚਲ dma_addr_t src_plane = octx->src[1]->ਅੰਕੜੇ + i12 * nb12 + i13 * nb13;
            ਜਦੋਂ (!dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(cur_dst, src_plane),
                                   ne10 * elem_ਮਾਪ, nb11, ne10 * elem_ਮਾਪ, ne11)) {
                dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_ਮਾਪ;
        }
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਪੂਰਨ op_get_ਪੰਕਤੀਆਂ(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params *) octx->kernel_params;

    ਅਚਲ uint32_t ne00          = octx->src[0]->ne[0];
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_ਮਾਪ = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(octx->src[0]->type, ne00);
    ਜੇ (src0_ਪੰਕਤੀ_ਮਾਪ == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (kparams->kernel_type == HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_SAMETYPE) {
        ਜੇ (octx->src[0]->type != octx->dst->type) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (octx->dst->type != HTP_TYPE_F32) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }

    ਜੇ (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਅਚਲ uint32_t ਕੁੱਲ_tasks    = kparams->ਕੁੱਲ_tasks;
    ਅਚਲ uint32_t dst_ਪੰਕਤੀ_ਮਾਪ   = (kparams->kernel_type == HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_SAMETYPE) ? src0_ਪੰਕਤੀ_ਮਾਪ : ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint32_t task_ਸ਼ੁਰੂ = 0;
    uint32_t tasks      = ਕੁੱਲ_tasks;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t tasks_ਪ੍ਰਤੀ_chunk = 1;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, dst_ਪੰਕਤੀ_ਮਾਪ / dst->ne[0], dst_ਪੰਕਤੀ_ਮਾਪ, &tasks_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_tasks, tasks_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        task_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        tasks      = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (tasks == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, (uint32_t) kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    ਬਣਤਰ get_ਪੰਕਤੀਆਂ_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_ਅਧਾਰ = (uint8_t *)octx->ctx->vtcm_ਅਧਾਰ;
    grctx.compute_flat = ਨਿਰਾ;
    grctx.compute_tiled = ਨਿਰਾ;
    grctx.task_ਸ਼ੁਰੂ = task_ਸ਼ੁਰੂ;
    grctx.tasks = tasks;
    grctx.tasks_ਪ੍ਰਤੀ_thread = octx->ctx->mdev.ਗਿਣਤੀ_2 == 1 ? kparams->tasks_ਪ੍ਰਤੀ_thread : fastdiv(tasks + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);
    grctx.tile_ਮਾਪ   = 0;
    grctx.tile_stride = 0;
    grctx.src0_ਪੰਕਤੀ_ਮਾਪ = src0_ਪੰਕਤੀ_ਮਾਪ;
    grctx.dst_ਪੰਕਤੀ_ਮਾਪ  = dst_ਪੰਕਤੀ_ਮਾਪ;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_ਪੰਕਤੀਆਂ_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, ਕੁੱਲ_tasks, grctx.idx_type, n_ਧਾਗੇ);

    ਜੇ (grctx.vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ > octx->ctx->vtcm_ਮਾਪ) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ਮਾਪ, grctx.vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    work_ਕਤਾਰ_func_t q_func = ਨਿਰਾ;
    ਚੋਣ (kparams->kernel_type) {
        ਹਾਲਤ HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_SAMETYPE:
            q_func = get_ਪੰਕਤੀਆਂ_thread_sametype;
            ਰੋਕੋ;
        ਹਾਲਤ HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_TILED:
            ਚੋਣ (octx->src[0]->type) {
                ਹਾਲਤ HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_f16_f32;  ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_q1_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_q2_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_q8_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_q4_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_q4_k_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_q5_k_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_ਪੰਕਤੀ_tiled_q6_k_f32; ਰੋਕੋ;
                ਮੂਲ:            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
            }
            grctx.tile_ਮਾਪ   = htp_tensor_get_tile_ਮਾਪ(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_ਮਾਪ(octx->src[0]->type);
            q_func = get_ਪੰਕਤੀਆਂ_thread_tiled;
            ਰੋਕੋ;
        ਹਾਲਤ HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_FLAT:
            ਚੋਣ (octx->src[0]->type) {
                ਹਾਲਤ HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_f16_f32;  ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_bf16_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_q1_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_q2_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_q8_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_q4_0_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_q4_k_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_q5_k_f32; ਰੋਕੋ;
                ਹਾਲਤ HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_ਪੰਕਤੀ_flat_q6_k_f32; ਰੋਕੋ;
                ਮੂਲ:            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
            }
            q_func = get_ਪੰਕਤੀਆਂ_thread_flat;
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t elem_ਮਾਪ = (grctx.idx_type == HTP_TYPE_I64) ? ਆਕਾਰ(int64_t) : ਆਕਾਰ(int32_t);
    ਅਚਲ size_t src1_ਬਾਈਟਾਂ  = (size_t) ਕੁੱਲ_tasks * elem_ਮਾਪ;

    ਜੇ (!htp_tensor_is_extended(octx->src[1]) && src1_ਬਾਈਟਾਂ <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_ਮਾਪ)) {
        grctx.src1_ਅੰਕੜੇ  = (ਅਚਲ ਖਾਲੀ *) (uintptr_t) octx->src[1]->ਅੰਕੜੇ;
        grctx.read_ਸੂਚਕਾਂਕ = get_ਪੰਕਤੀਆਂ_read_ਸੂਚਕਾਂਕ_ddr;
    } ਨਹੀਂਤਾਂ {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_ਅਧਾਰ, grctx.vtcm_layout.off_src1);
        get_ਪੰਕਤੀਆਂ_dma_src1(octx, vtcm_src1, elem_ਮਾਪ);
        grctx.src1_ਅੰਕੜੇ  = vtcm_src1;
        grctx.read_ਸੂਚਕਾਂਕ = get_ਪੰਕਤੀਆਂ_read_ਸੂਚਕਾਂਕ_vtcm;
    }

    FARF(ਉੱਚਾ_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ,
         grctx.vtcm_layout.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  * n_ਧਾਗੇ,
         kparams->kernel_type, n_ਧਾਗੇ);

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, q_func, &grctx, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
