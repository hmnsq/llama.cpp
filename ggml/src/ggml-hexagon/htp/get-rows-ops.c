#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-quant.h"
#ಅನ್ತರ್ಭಾವ "get-rows-ops.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"

ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint32_t (*get_ಅಡ್ಡಸಾಲುಗಳು_read_ಸೂಚ್ಯಂಕ_fn_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1, uint32_t idx, ಪೂರ್ಣಾಂಕ idx_type);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*get_ಅಡ್ಡಸಾಲುಗಳು_flat_compute_fn_t)(ದಶಮಾಂಶ * dst_spad, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src_spad, uint32_t cur_elems);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*get_ಅಡ್ಡಸಾಲುಗಳು_tiled_compute_fn_t)(ದಶಮಾಂಶ * dst_spad, ಸ್ಥಿರಾಂಕ uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t ಅಡ್ಡಸಾಲು);

ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams;
    ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_ತಳ;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1_ದತ್ತಾಂಶ;
    get_ಅಡ್ಡಸಾಲುಗಳು_read_ಸೂಚ್ಯಂಕ_fn_t    read_ಸೂಚ್ಯಂಕ;
    get_ಅಡ್ಡಸಾಲುಗಳು_flat_compute_fn_t  compute_flat;
    get_ಅಡ್ಡಸಾಲುಗಳು_tiled_compute_fn_t compute_tiled;
    uint32_t task_ಆರಂಭ;
    uint32_t tasks;
    uint32_t tasks_ಪ್ರತಿ_thread;
    uint32_t tile_ಗಾತ್ರ;
    uint32_t tile_stride;
    uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    uint32_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ idx_type;
};

#ವ್ಯಾಖ್ಯಾನಿಸು get_ಅಡ್ಡಸಾಲುಗಳು_preamble                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = octx->src[0]->ne[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = octx->src[0]->ne[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = octx->src[0]->ne[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = octx->src[1]->ne[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne11 = octx->src[1]->ne[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = octx->src[1]->ne[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = octx->dst->ne[0];     \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = octx->dst->ne[1];     \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = octx->dst->ne[2];     \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = octx->src[0]->nb[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = octx->src[0]->nb[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nb10 = octx->src[1]->nb[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = octx->src[1]->nb[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = octx->dst->nb[1];     \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = octx->dst->nb[2];     \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nr = ne10 * ne11 * ne12;

ಸ್ಥಿರ uint32_t get_ಅಡ್ಡಸಾಲುಗಳು_read_ಸೂಚ್ಯಂಕ_ddr(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1,
    uint32_t idx,
    ಪೂರ್ಣಾಂಕ idx_type
) {
    ಸ್ಥಿರಾಂಕ uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? ಗಾತ್ರ(int64_t) : ಗಾತ್ರ(int32_t));
    ಹಿಂತಿರುಗಿ *(ಸ್ಥಿರಾಂಕ uint32_t *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src1 + off);
}

ಸ್ಥಿರ __attribute__((noinline)) uint32_t get_ಅಡ್ಡಸಾಲುಗಳು_read_ಸೂಚ್ಯಂಕ_vtcm(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1,
    uint32_t idx,
    ಪೂರ್ಣಾಂಕ idx_type
) {
    ಸ್ಥಿರಾಂಕ uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? ಗಾತ್ರ(int64_t) : ಗಾತ್ರ(int32_t));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v = hvx_vmem((ಸ್ಥಿರಾಂಕ uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    ಹಿಂತಿರುಗಿ (uint32_t) Q6_R_vextract_VR(v, off);
}

ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_thread_sametype(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context * grctx = (ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = grctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams = grctx->kparams;
    get_ಅಡ್ಡಸಾಲುಗಳು_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t dr  = grctx->tasks_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = grctx->task_ಆರಂಭ + dr * ith;
    ಆದರೆ (ir0 >= grctx->task_ಆರಂಭ + grctx->tasks) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, grctx->task_ಆರಂಭ + grctx->tasks);
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = grctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಗಾಗಿ (uint32_t i = ir0; i < ir1; ++i) {
        ಸ್ಥಿರಾಂಕ uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        ಸ್ಥಿರಾಂಕ uint32_t rem = i - i12 * ne11 * ne10;
        ಸ್ಥಿರಾಂಕ uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        ಸ್ಥಿರಾಂಕ uint32_t i10 = rem - i11 * ne10;

        ಸ್ಥಿರಾಂಕ uint32_t i01 = grctx->read_ಸೂಚ್ಯಂಕ(grctx->src1_ದತ್ತಾಂಶ, i, grctx->idx_type);
        ಪ್ರತಿಜ್ಞಾ(i01 < ne01);

        ಸ್ಥಿರಾಂಕ uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ಸ್ಥಿರಾಂಕ uint32_t i02 = i11 - q02 * ne02;
        ಸ್ಥಿರಾಂಕ uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ಸ್ಥಿರಾಂಕ uint32_t i03 = i12 - q03 * ne03;

        ಸ್ಥಿರಾಂಕ dma_addr_t src0_ದತ್ತಾಂಶ = octx->src[0]->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;
        ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ  = octx->dst->ದತ್ತಾಂಶ    + i10*nb1  + i11*nb2  + i12*nb3;

        ಆಗ (!dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, src0_ದತ್ತಾಂಶ), nb1, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, 1)) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }
    }
    dma_ಸರತಿ_flush(dma_q);
}



ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_flat_task {
    dma_addr_t src0_ದತ್ತಾಂಶ;
    dma_addr_t dst_ದತ್ತಾಂಶ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_flat_task get_ಅಡ್ಡಸಾಲುಗಳು_flat_calc_task(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context * grctx,
    uint32_t i,
    bool need_src0
) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams = grctx->kparams;
    get_ಅಡ್ಡಸಾಲುಗಳು_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ಸ್ಥಿರಾಂಕ uint32_t rem = i - i12 * ne11 * ne10;
    ಸ್ಥಿರಾಂಕ uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ಸ್ಥಿರಾಂಕ uint32_t i10 = rem - i11 * ne10;

    ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_flat_task task;
    task.src0_ದತ್ತಾಂಶ = 0;
    task.dst_ದತ್ತಾಂಶ  = octx->dst->ದತ್ತಾಂಶ + i10*nb1 + i11*nb2 + i12*nb3;

    ಆದರೆ (need_src0) {
        ಸ್ಥಿರಾಂಕ uint32_t i01 = grctx->read_ಸೂಚ್ಯಂಕ(grctx->src1_ದತ್ತಾಂಶ, i, grctx->idx_type);
        ಪ್ರತಿಜ್ಞಾ(i01 < ne01);

        ಸ್ಥಿರಾಂಕ uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        ಸ್ಥಿರಾಂಕ uint32_t i02 = i11 - q02 * ne02;
        ಸ್ಥಿರಾಂಕ uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        ಸ್ಥಿರಾಂಕ uint32_t i03 = i12 - q03 * ne03;

        task.src0_ದತ್ತಾಂಶ = octx->src[0]->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;
    }

    ಹಿಂತಿರುಗಿ task;
}

ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_thread_flat(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context * grctx = (ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = grctx->octx;
    get_ಅಡ್ಡಸಾಲುಗಳು_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t dr  = grctx->tasks_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = grctx->task_ಆರಂಭ + dr * ith;
    ಆದರೆ (ir0 >= grctx->task_ಆರಂಭ + grctx->tasks) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, grctx->task_ಆರಂಭ + grctx->tasks);

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ತಳ, vtcm_layout->off_src0) + ith * vtcm_layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ತಳ, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಬೈಟುಗಳು = grctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಬೈಟುಗಳು  = grctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    get_ಅಡ್ಡಸಾಲುಗಳು_flat_compute_fn_t compute = grctx->compute_flat;

    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0, spad_idx = 0; ಹೆಜ್ಜೆ < ir1 - ir0 && spad_idx < 2; ++ಹೆಜ್ಜೆ, spad_idx++) {
        ಸ್ಥಿರಾಂಕ uint32_t i = ir0 + ಹೆಜ್ಜೆ;
        ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_flat_task task = get_ಅಡ್ಡಸಾಲುಗಳು_flat_calc_task(octx, grctx, i, true);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(octx->dst->ದತ್ತಾಂಶ,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_ಗಾತ್ರ),
                       dst_ಬೈಟುಗಳು, vtcm_layout->dst_spad_half_ಗಾತ್ರ, dst_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_ಗಾತ್ರ, task.src0_ದತ್ತಾಂಶ),
                       vtcm_layout->src0_spad_half_ಗಾತ್ರ, src0_ಬೈಟುಗಳು, src0_ಬೈಟುಗಳು, 1);
    }

    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < ir1 - ir0; ++ಹೆಜ್ಜೆ) {
        ಸ್ಥಿರಾಂಕ uint32_t i = ir0 + ಹೆಜ್ಜೆ;
        ಶೂನ್ಯ * dst_spad = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * src_spad = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_flat_task task = get_ಅಡ್ಡಸಾಲುಗಳು_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((ದಶಮಾಂಶ *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(task.dst_ದತ್ತಾಂಶ, dst_spad),
                       dst_ಬೈಟುಗಳು, vtcm_layout->dst_spad_half_ಗಾತ್ರ, dst_ಬೈಟುಗಳು, 1);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಹೆಜ್ಜೆ = ಹೆಜ್ಜೆ + 2;
        ಆದರೆ (ಮುಂದಿನ_ಹೆಜ್ಜೆ < ir1 - ir0) {
            ಸ್ಥಿರಾಂಕ uint32_t pi = ir0 + ಮುಂದಿನ_ಹೆಜ್ಜೆ;
            ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_flat_task ಮುಂದಿನ_task = get_ಅಡ್ಡಸಾಲುಗಳು_flat_calc_task(octx, grctx, pi, true);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                           dma_make_ದತ್ತಾಂಶ(src_spad, ಮುಂದಿನ_task.src0_ದತ್ತಾಂಶ),
                           vtcm_layout->src0_spad_half_ಗಾತ್ರ, src0_ಬೈಟುಗಳು, src0_ಬೈಟುಗಳು, 1);
        }
    }

    dma_ಸರತಿ_flush(dma_q);
}



ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_tiled_task {
    dma_addr_t tile_src_ತಳ;
    dma_addr_t dst_ದತ್ತಾಂಶ;
    uint32_t   ಅಡ್ಡಸಾಲು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_tiled_task get_ಅಡ್ಡಸಾಲುಗಳು_tiled_calc_task(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_ಗಾತ್ರ
) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams = grctx->kparams;
    get_ಅಡ್ಡಸಾಲುಗಳು_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    ಸ್ಥಿರಾಂಕ uint32_t rem = i - i12 * ne11 * ne10;
    ಸ್ಥಿರಾಂಕ uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    ಸ್ಥಿರಾಂಕ uint32_t i10 = rem - i11 * ne10;
    ಸ್ಥಿರಾಂಕ uint32_t i01 = grctx->read_ಸೂಚ್ಯಂಕ(grctx->src1_ದತ್ತಾಂಶ, i, grctx->idx_type);
    ಪ್ರತಿಜ್ಞಾ(i01 < ne01);

    ಸ್ಥಿರಾಂಕ uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    ಸ್ಥಿರಾಂಕ uint32_t i02 = i11 - q02 * ne02;
    ಸ್ಥಿರಾಂಕ uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    ಸ್ಥಿರಾಂಕ uint32_t i03 = i12 - q03 * ne03;
    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ತಂಭ_tile = i01 / HTP_TENSOR_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು = i01 % HTP_TENSOR_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ dma_addr_t ಮಾತೃಕೆ = octx->src[0]->ದತ್ತಾಂಶ + i02*nb02 + i03*nb03;

    ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_tiled_task task;
    task.tile_src_ತಳ = ಮಾತೃಕೆ + (ಸ್ತಂಭ_tile * n_k_tiles) * tile_ಗಾತ್ರ;
    task.dst_ದತ್ತಾಂಶ      = octx->dst->ದತ್ತಾಂಶ + i10*nb1 + i11*nb2 + i12*nb3;
    task.ಅಡ್ಡಸಾಲು           = ಅಡ್ಡಸಾಲು;
    ಹಿಂತಿರುಗಿ task;
}

ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_thread_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context * grctx = (ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = grctx->octx;
    get_ಅಡ್ಡಸಾಲುಗಳು_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t dr  = grctx->tasks_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = grctx->task_ಆರಂಭ + dr * ith;
    ಆದರೆ (ir0 >= grctx->task_ಆರಂಭ + grctx->tasks) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, grctx->task_ಆರಂಭ + grctx->tasks);
    ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ತಳ, vtcm_layout->off_src0) + ith * vtcm_layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint8_t * dst_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_ತಳ, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ uint32_t tile_ಗಾತ್ರ   = grctx->tile_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t tile_stride = grctx->tile_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಬೈಟುಗಳು   = grctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    get_ಅಡ್ಡಸಾಲುಗಳು_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0, spad_idx = 0; ಹೆಜ್ಜೆ < ir1 - ir0 && spad_idx < 2; ++ಹೆಜ್ಜೆ, ++spad_idx) {
        ಸ್ಥಿರಾಂಕ uint32_t i = ir0 + ಹೆಜ್ಜೆ;
        ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_tiled_task task = get_ಅಡ್ಡಸಾಲುಗಳು_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_ಗಾತ್ರ);

        // Dummy writeback to prime the queue with dst descriptor
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(task.dst_ದತ್ತಾಂಶ, dst_spad_ತಳ + spad_idx * vtcm_layout->dst_spad_half_ಗಾತ್ರ),
                       dst_ಬೈಟುಗಳು, vtcm_layout->dst_spad_half_ಗಾತ್ರ, dst_ಬೈಟುಗಳು, 0);

        // Prefetch row tiles
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(src_spad_ತಳ + spad_idx * vtcm_layout->src0_spad_half_ಗಾತ್ರ, task.tile_src_ತಳ),
                       tile_stride, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles);
    }

    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < ir1 - ir0; ++ಹೆಜ್ಜೆ) {
        ಸ್ಥಿರಾಂಕ uint32_t i = ir0 + ಹೆಜ್ಜೆ;
        ದಶಮಾಂಶ * dst_spad   = (ದಶಮಾಂಶ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_tiled_task task = get_ಅಡ್ಡಸಾಲುಗಳು_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_ಗಾತ್ರ);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.ಅಡ್ಡಸಾಲು);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(task.dst_ದತ್ತಾಂಶ, dst_spad),
                       dst_ಬೈಟುಗಳು, vtcm_layout->dst_spad_half_ಗಾತ್ರ, dst_ಬೈಟುಗಳು, 1);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಹೆಜ್ಜೆ = ಹೆಜ್ಜೆ + 2;
        ಆದರೆ (ಮುಂದಿನ_ಹೆಜ್ಜೆ < ir1 - ir0) {
            ಸ್ಥಿರಾಂಕ uint32_t ni = ir0 + ಮುಂದಿನ_ಹೆಜ್ಜೆ;
            ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_tiled_task ಮುಂದಿನ_task = get_ಅಡ್ಡಸಾಲುಗಳು_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_ಗಾತ್ರ);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                           dma_make_ದತ್ತಾಂಶ(src_spad, ಮುಂದಿನ_task.tile_src_ತಳ),
                           tile_stride, tile_ಗಾತ್ರ, tile_ಗಾತ್ರ, n_k_tiles);
        }
    }

    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_dma_src1(
    ರಚನೆ htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_ಗಾತ್ರ
) {
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = octx->src[1]->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne11 = octx->src[1]->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = octx->src[1]->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = octx->src[1]->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = octx->src[1]->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = octx->src[1]->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = octx->src[1]->nb[3];

    dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    ಗಾಗಿ (uint32_t i13 = 0; i13 < ne13; ++i13) {
        ಗಾಗಿ (uint32_t i12 = 0; i12 < ne12; ++i12) {
            ಸ್ಥಿರಾಂಕ dma_addr_t src_plane = octx->src[1]->ದತ್ತಾಂಶ + i12 * nb12 + i13 * nb13;
            ಆಗ (!dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(cur_dst, src_plane),
                                   ne10 * elem_ಗಾತ್ರ, nb11, ne10 * elem_ಗಾತ್ರ, ne11)) {
                dma_ಸರತಿ_ತೆಗೆ(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_ಗಾತ್ರ;
        }
    }
    dma_ಸರತಿ_flush(dma_q);
}

ಪೂರ್ಣಾಂಕ op_get_ಅಡ್ಡಸಾಲುಗಳು(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params *) octx->kernel_params;

    ಸ್ಥಿರಾಂಕ uint32_t ne00          = octx->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(octx->src[0]->type, ne00);
    ಆದರೆ (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (kparams->kernel_type == HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_SAMETYPE) {
        ಆದರೆ (octx->src[0]->type != octx->dst->type) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (octx->dst->type != HTP_TYPE_F32) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }

    ಆದರೆ (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_tasks    = kparams->ಒಟ್ಟು_tasks;
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ   = (kparams->kernel_type == HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_SAMETYPE) ? src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ : ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint32_t task_ಆರಂಭ = 0;
    uint32_t tasks      = ಒಟ್ಟು_tasks;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t tasks_ಪ್ರತಿ_chunk = 1;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ / dst->ne[0], dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &tasks_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_tasks, tasks_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        task_ಆರಂಭ = range.ಆರಂಭ;
        tasks      = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (tasks == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, (uint32_t) kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    ರಚನೆ get_ಅಡ್ಡಸಾಲುಗಳು_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_ತಳ = (uint8_t *)octx->ctx->vtcm_ತಳ;
    grctx.compute_flat = ಖಾಲಿ;
    grctx.compute_tiled = ಖಾಲಿ;
    grctx.task_ಆರಂಭ = task_ಆರಂಭ;
    grctx.tasks = tasks;
    grctx.tasks_ಪ್ರತಿ_thread = octx->ctx->mdev.ಎಣಿಕೆ_2 == 1 ? kparams->tasks_ಪ್ರತಿ_thread : fastdiv(tasks + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);
    grctx.tile_ಗಾತ್ರ   = 0;
    grctx.tile_stride = 0;
    grctx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    grctx.dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, ಒಟ್ಟು_tasks, grctx.idx_type, n_ಎಳೆಗಳು);

    ಆದರೆ (grctx.vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು > octx->ctx->vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_ಗಾತ್ರ, grctx.vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    work_ಸರತಿ_func_t q_func = ಖಾಲಿ;
    ಆಯ್ಕೆ (kparams->kernel_type) {
        ಸ್ಥಿತಿ HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_SAMETYPE:
            q_func = get_ಅಡ್ಡಸಾಲುಗಳು_thread_sametype;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_TILED:
            ಆಯ್ಕೆ (octx->src[0]->type) {
                ಸ್ಥಿತಿ HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_f16_f32;  ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_q1_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_q2_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_q8_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_q4_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_q4_k_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_q5_k_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_ಅಡ್ಡಸಾಲು_tiled_q6_k_f32; ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            }
            grctx.tile_ಗಾತ್ರ   = htp_tensor_get_tile_ಗಾತ್ರ(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_ಗಾತ್ರ(octx->src[0]->type);
            q_func = get_ಅಡ್ಡಸಾಲುಗಳು_thread_tiled;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_FLAT:
            ಆಯ್ಕೆ (octx->src[0]->type) {
                ಸ್ಥಿತಿ HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_f16_f32;  ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_bf16_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q1_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q2_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q8_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q4_0_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q4_k_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q5_k_f32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q6_k_f32; ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            }
            q_func = get_ಅಡ್ಡಸಾಲುಗಳು_thread_flat;
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = (grctx.idx_type == HTP_TYPE_I64) ? ಗಾತ್ರ(int64_t) : ಗಾತ್ರ(int32_t);
    ಸ್ಥಿರಾಂಕ size_t src1_ಬೈಟುಗಳು  = (size_t) ಒಟ್ಟು_tasks * elem_ಗಾತ್ರ;

    ಆದರೆ (!htp_tensor_is_extended(octx->src[1]) && src1_ಬೈಟುಗಳು <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_ಗಾತ್ರ)) {
        grctx.src1_ದತ್ತಾಂಶ  = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) (uintptr_t) octx->src[1]->ದತ್ತಾಂಶ;
        grctx.read_ಸೂಚ್ಯಂಕ = get_ಅಡ್ಡಸಾಲುಗಳು_read_ಸೂಚ್ಯಂಕ_ddr;
    } ಇಲ್ಲದಿದ್ದರೆ {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_ತಳ, grctx.vtcm_layout.off_src1);
        get_ಅಡ್ಡಸಾಲುಗಳು_dma_src1(octx, vtcm_src1, elem_ಗಾತ್ರ);
        grctx.src1_ದತ್ತಾಂಶ  = vtcm_src1;
        grctx.read_ಸೂಚ್ಯಂಕ = get_ಅಡ್ಡಸಾಲುಗಳು_read_ಸೂಚ್ಯಂಕ_vtcm;
    }

    FARF(ಮೇಲಿನ_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು,
         grctx.vtcm_layout.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  * n_ಎಳೆಗಳು,
         kparams->kernel_type, n_ಎಳೆಗಳು);

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, q_func, &grctx, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
