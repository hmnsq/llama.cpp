#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hvx-quant.h"
#అన్తర్భావ "get-rows-ops.h"
#అన్తర్భావ "work-queue.h"

నిర్మాణం get_అడ్డువరుసలు_context;

రకనిర్వచనం uint32_t (*get_అడ్డువరుసలు_read_సూచీ_fn_t)(స్థిరరాశి శూన్య * src1, uint32_t idx, పూర్ణాంకం idx_type);
రకనిర్వచనం శూన్య (*get_అడ్డువరుసలు_flat_compute_fn_t)(దశాంశం * dst_spad, స్థిరరాశి శూన్య * src_spad, uint32_t cur_elems);
రకనిర్వచనం శూన్య (*get_అడ్డువరుసలు_tiled_compute_fn_t)(దశాంశం * dst_spad, స్థిరరాశి uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t అడ్డువరుస);

నిర్మాణం get_అడ్డువరుసలు_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params * kparams;
    నిర్మాణం htp_get_అడ్డువరుసలు_vtcm_layout vtcm_layout;
    uint8_t    * vtcm_పునాది;
    స్థిరరాశి శూన్య * src1_దత్తాంశం;
    get_అడ్డువరుసలు_read_సూచీ_fn_t    read_సూచీ;
    get_అడ్డువరుసలు_flat_compute_fn_t  compute_flat;
    get_అడ్డువరుసలు_tiled_compute_fn_t compute_tiled;
    uint32_t task_ప్రారంభం;
    uint32_t tasks;
    uint32_t tasks_చొప్పున_thread;
    uint32_t tile_కొలత;
    uint32_t tile_stride;
    uint32_t src0_అడ్డువరుస_కొలత;
    uint32_t dst_అడ్డువరుస_కొలత;
    పూర్ణాంకం idx_type;
};

#నిర్వచించు get_అడ్డువరుసలు_preamble                      \
    స్థిరరాశి uint32_t ne00 = octx->src[0]->ne[0]; \
    స్థిరరాశి uint32_t ne01 = octx->src[0]->ne[1]; \
    స్థిరరాశి uint32_t ne02 = octx->src[0]->ne[2]; \
    స్థిరరాశి uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    స్థిరరాశి uint32_t ne10 = octx->src[1]->ne[0]; \
    స్థిరరాశి uint32_t ne11 = octx->src[1]->ne[1]; \
    స్థిరరాశి uint32_t ne12 = octx->src[1]->ne[2]; \
    స్థిరరాశి uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    స్థిరరాశి uint32_t ne0 = octx->dst->ne[0];     \
    స్థిరరాశి uint32_t ne1 = octx->dst->ne[1];     \
    స్థిరరాశి uint32_t ne2 = octx->dst->ne[2];     \
    స్థిరరాశి uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    స్థిరరాశి uint32_t nb01 = octx->src[0]->nb[1]; \
    స్థిరరాశి uint32_t nb02 = octx->src[0]->nb[2]; \
    స్థిరరాశి uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    స్థిరరాశి uint32_t nb10 = octx->src[1]->nb[0]; \
    స్థిరరాశి uint32_t nb11 = octx->src[1]->nb[1]; \
    స్థిరరాశి uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    స్థిరరాశి uint32_t nb1 = octx->dst->nb[1];     \
    స్థిరరాశి uint32_t nb2 = octx->dst->nb[2];     \
    స్థిరరాశి uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    స్థిరరాశి uint32_t nr = ne10 * ne11 * ne12;

స్థిర uint32_t get_అడ్డువరుసలు_read_సూచీ_ddr(
    స్థిరరాశి శూన్య * src1,
    uint32_t idx,
    పూర్ణాంకం idx_type
) {
    స్థిరరాశి uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? పరిమాణం(int64_t) : పరిమాణం(int32_t));
    తిరిగి *(స్థిరరాశి uint32_t *) ((స్థిరరాశి uint8_t *) src1 + off);
}

స్థిర __attribute__((noinline)) uint32_t get_అడ్డువరుసలు_read_సూచీ_vtcm(
    స్థిరరాశి శూన్య * src1,
    uint32_t idx,
    పూర్ణాంకం idx_type
) {
    స్థిరరాశి uint32_t off = idx * ((idx_type == HTP_TYPE_I64) ? పరిమాణం(int64_t) : పరిమాణం(int32_t));
    స్థిరరాశి HVX_సదిశ v = hvx_vmem((స్థిరరాశి uint8_t *) src1 + off); // off & ~127u in HW
    // vextract is normally slow, but here it avoids an HVX->L2 store and L1D miss
    తిరిగి (uint32_t) Q6_R_vextract_VR(v, off);
}

స్థిర శూన్య get_అడ్డువరుసలు_thread_sametype(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం get_అడ్డువరుసలు_context * grctx = (నిర్మాణం get_అడ్డువరుసలు_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = grctx->octx;
    స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params * kparams = grctx->kparams;
    get_అడ్డువరుసలు_preamble;

    స్థిరరాశి uint32_t dr  = grctx->tasks_చొప్పున_thread;
    స్థిరరాశి uint32_t ir0 = grctx->task_ప్రారంభం + dr * ith;
    ఒకవేళ (ir0 >= grctx->task_ప్రారంభం + grctx->tasks) {
        తిరిగి;
    }
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, grctx->task_ప్రారంభం + grctx->tasks);
    స్థిరరాశి uint32_t అడ్డువరుస_కొలత_బైట్లు = grctx->src0_అడ్డువరుస_కొలత;
    dma_వరుస * dma_q = octx->ctx->dma[ith];

    కొరకు (uint32_t i = ir0; i < ir1; ++i) {
        స్థిరరాశి uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
        స్థిరరాశి uint32_t rem = i - i12 * ne11 * ne10;
        స్థిరరాశి uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
        స్థిరరాశి uint32_t i10 = rem - i11 * ne10;

        స్థిరరాశి uint32_t i01 = grctx->read_సూచీ(grctx->src1_దత్తాంశం, i, grctx->idx_type);
        ప్రతిజ్ఞా(i01 < ne01);

        స్థిరరాశి uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        స్థిరరాశి uint32_t i02 = i11 - q02 * ne02;
        స్థిరరాశి uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        స్థిరరాశి uint32_t i03 = i12 - q03 * ne03;

        స్థిరరాశి dma_addr_t src0_దత్తాంశం = octx->src[0]->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;
        స్థిరరాశి dma_addr_t dst_దత్తాంశం  = octx->dst->దత్తాంశం    + i10*nb1  + i11*nb2  + i12*nb3;

        అయితే (!dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_దత్తాంశం, src0_దత్తాంశం), nb1, nb01, అడ్డువరుస_కొలత_బైట్లు, 1)) {
            dma_వరుస_తీయి(dma_q);
        }
    }
    dma_వరుస_flush(dma_q);
}



నిర్మాణం get_అడ్డువరుసలు_flat_task {
    dma_addr_t src0_దత్తాంశం;
    dma_addr_t dst_దత్తాంశం;
};

స్థిర అంతఃవరుస నిర్మాణం get_అడ్డువరుసలు_flat_task get_అడ్డువరుసలు_flat_calc_task(
    స్థిరరాశి నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం get_అడ్డువరుసలు_context * grctx,
    uint32_t i,
    bool need_src0
) {
    స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params * kparams = grctx->kparams;
    get_అడ్డువరుసలు_preamble;

    స్థిరరాశి uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    స్థిరరాశి uint32_t rem = i - i12 * ne11 * ne10;
    స్థిరరాశి uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    స్థిరరాశి uint32_t i10 = rem - i11 * ne10;

    నిర్మాణం get_అడ్డువరుసలు_flat_task task;
    task.src0_దత్తాంశం = 0;
    task.dst_దత్తాంశం  = octx->dst->దత్తాంశం + i10*nb1 + i11*nb2 + i12*nb3;

    ఒకవేళ (need_src0) {
        స్థిరరాశి uint32_t i01 = grctx->read_సూచీ(grctx->src1_దత్తాంశం, i, grctx->idx_type);
        ప్రతిజ్ఞా(i01 < ne01);

        స్థిరరాశి uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
        స్థిరరాశి uint32_t i02 = i11 - q02 * ne02;
        స్థిరరాశి uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
        స్థిరరాశి uint32_t i03 = i12 - q03 * ne03;

        task.src0_దత్తాంశం = octx->src[0]->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;
    }

    తిరిగి task;
}

స్థిర శూన్య get_అడ్డువరుసలు_thread_flat(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం get_అడ్డువరుసలు_context * grctx = (నిర్మాణం get_అడ్డువరుసలు_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = grctx->octx;
    get_అడ్డువరుసలు_preamble;

    స్థిరరాశి uint32_t dr  = grctx->tasks_చొప్పున_thread;
    స్థిరరాశి uint32_t ir0 = grctx->task_ప్రారంభం + dr * ith;
    ఒకవేళ (ir0 >= grctx->task_ప్రారంభం + grctx->tasks) {
        తిరిగి;
    }
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, grctx->task_ప్రారంభం + grctx->tasks);

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_పునాది, vtcm_layout->off_src0) + ith * vtcm_layout->src0_బైట్లు_చొప్పున_thread;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_పునాది, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_బైట్లు_చొప్పున_thread;
    స్థిరరాశి uint32_t src0_బైట్లు = grctx->src0_అడ్డువరుస_కొలత;
    స్థిరరాశి uint32_t dst_బైట్లు  = grctx->dst_అడ్డువరుస_కొలత;
    get_అడ్డువరుసలు_flat_compute_fn_t compute = grctx->compute_flat;

    కొరకు (uint32_t అడుగు = 0, spad_idx = 0; అడుగు < ir1 - ir0 && spad_idx < 2; ++అడుగు, spad_idx++) {
        స్థిరరాశి uint32_t i = ir0 + అడుగు;
        నిర్మాణం get_అడ్డువరుసలు_flat_task task = get_అడ్డువరుసలు_flat_calc_task(octx, grctx, i, true);

        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(octx->dst->దత్తాంశం,
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_కొలత),
                       dst_బైట్లు, vtcm_layout->dst_spad_half_కొలత, dst_బైట్లు, 0);
        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_కొలత, task.src0_దత్తాంశం),
                       vtcm_layout->src0_spad_half_కొలత, src0_బైట్లు, src0_బైట్లు, 1);
    }

    కొరకు (uint32_t అడుగు = 0; అడుగు < ir1 - ir0; ++అడుగు) {
        స్థిరరాశి uint32_t i = ir0 + అడుగు;
        శూన్య * dst_spad = (శూన్య *) dma_వరుస_తీయి(dma_q).src;
        శూన్య * src_spad = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

        నిర్మాణం get_అడ్డువరుసలు_flat_task task = get_అడ్డువరుసలు_flat_calc_task(octx, grctx, i, false);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute((దశాంశం *) dst_spad, src_spad, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(task.dst_దత్తాంశం, dst_spad),
                       dst_బైట్లు, vtcm_layout->dst_spad_half_కొలత, dst_బైట్లు, 1);

        స్థిరరాశి uint32_t తదుపరి_అడుగు = అడుగు + 2;
        ఒకవేళ (తదుపరి_అడుగు < ir1 - ir0) {
            స్థిరరాశి uint32_t pi = ir0 + తదుపరి_అడుగు;
            నిర్మాణం get_అడ్డువరుసలు_flat_task తదుపరి_task = get_అడ్డువరుసలు_flat_calc_task(octx, grctx, pi, true);
            dma_వరుస_నెట్టు(dma_q,
                           dma_make_దత్తాంశం(src_spad, తదుపరి_task.src0_దత్తాంశం),
                           vtcm_layout->src0_spad_half_కొలత, src0_బైట్లు, src0_బైట్లు, 1);
        }
    }

    dma_వరుస_flush(dma_q);
}



నిర్మాణం get_అడ్డువరుసలు_tiled_task {
    dma_addr_t tile_src_పునాది;
    dma_addr_t dst_దత్తాంశం;
    uint32_t   అడ్డువరుస;
};

స్థిర అంతఃవరుస నిర్మాణం get_అడ్డువరుసలు_tiled_task get_అడ్డువరుసలు_tiled_calc_task(
    స్థిరరాశి నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం get_అడ్డువరుసలు_context * grctx,
    uint32_t i,
    uint32_t n_k_tiles,
    uint32_t tile_కొలత
) {
    స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params * kparams = grctx->kparams;
    get_అడ్డువరుసలు_preamble;

    స్థిరరాశి uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);
    స్థిరరాశి uint32_t rem = i - i12 * ne11 * ne10;
    స్థిరరాశి uint32_t i11 = fastdiv(rem, &kparams->div_ne10);
    స్థిరరాశి uint32_t i10 = rem - i11 * ne10;
    స్థిరరాశి uint32_t i01 = grctx->read_సూచీ(grctx->src1_దత్తాంశం, i, grctx->idx_type);
    ప్రతిజ్ఞా(i01 < ne01);

    స్థిరరాశి uint32_t q02 = fastdiv(i11, &kparams->div_ne02);
    స్థిరరాశి uint32_t i02 = i11 - q02 * ne02;
    స్థిరరాశి uint32_t q03 = fastdiv(i12, &kparams->div_ne03);
    స్థిరరాశి uint32_t i03 = i12 - q03 * ne03;
    స్థిరరాశి uint32_t స్తంభం_tile = i01 / HTP_TENSOR_TILE_N_అడ్డువరుసలు;
    స్థిరరాశి uint32_t అడ్డువరుస = i01 % HTP_TENSOR_TILE_N_అడ్డువరుసలు;
    స్థిరరాశి dma_addr_t మాతృక = octx->src[0]->దత్తాంశం + i02*nb02 + i03*nb03;

    నిర్మాణం get_అడ్డువరుసలు_tiled_task task;
    task.tile_src_పునాది = మాతృక + (స్తంభం_tile * n_k_tiles) * tile_కొలత;
    task.dst_దత్తాంశం      = octx->dst->దత్తాంశం + i10*nb1 + i11*nb2 + i12*nb3;
    task.అడ్డువరుస           = అడ్డువరుస;
    తిరిగి task;
}

స్థిర శూన్య get_అడ్డువరుసలు_thread_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం get_అడ్డువరుసలు_context * grctx = (నిర్మాణం get_అడ్డువరుసలు_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = grctx->octx;
    get_అడ్డువరుసలు_preamble;

    స్థిరరాశి uint32_t dr  = grctx->tasks_చొప్పున_thread;
    స్థిరరాశి uint32_t ir0 = grctx->task_ప్రారంభం + dr * ith;
    ఒకవేళ (ir0 >= grctx->task_ప్రారంభం + grctx->tasks) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, grctx->task_ప్రారంభం + grctx->tasks);
    స్థిరరాశి uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(octx->src[0]->type);
    స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;
    uint8_t * src_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_పునాది, vtcm_layout->off_src0) + ith * vtcm_layout->src0_బైట్లు_చొప్పున_thread;
    uint8_t * dst_spad_పునాది = VTCM_LAYOUT_PTR(uint8_t, grctx->vtcm_పునాది, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_బైట్లు_చొప్పున_thread;
    dma_వరుస * dma_q = octx->ctx->dma[ith];
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి uint32_t tile_కొలత   = grctx->tile_కొలత;
    స్థిరరాశి uint32_t tile_stride = grctx->tile_stride;
    స్థిరరాశి uint32_t dst_బైట్లు   = grctx->dst_అడ్డువరుస_కొలత;
    get_అడ్డువరుసలు_tiled_compute_fn_t compute_tiled = grctx->compute_tiled;

    కొరకు (uint32_t అడుగు = 0, spad_idx = 0; అడుగు < ir1 - ir0 && spad_idx < 2; ++అడుగు, ++spad_idx) {
        స్థిరరాశి uint32_t i = ir0 + అడుగు;
        నిర్మాణం get_అడ్డువరుసలు_tiled_task task = get_అడ్డువరుసలు_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_కొలత);

        // Dummy writeback to prime the queue with dst descriptor
        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(task.dst_దత్తాంశం, dst_spad_పునాది + spad_idx * vtcm_layout->dst_spad_half_కొలత),
                       dst_బైట్లు, vtcm_layout->dst_spad_half_కొలత, dst_బైట్లు, 0);

        // Prefetch row tiles
        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(src_spad_పునాది + spad_idx * vtcm_layout->src0_spad_half_కొలత, task.tile_src_పునాది),
                       tile_stride, tile_కొలత, tile_కొలత, n_k_tiles);
    }

    కొరకు (uint32_t అడుగు = 0; అడుగు < ir1 - ir0; ++అడుగు) {
        స్థిరరాశి uint32_t i = ir0 + అడుగు;
        దశాంశం * dst_spad   = (దశాంశం *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * src_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        నిర్మాణం get_అడ్డువరుసలు_tiled_task task = get_అడ్డువరుసలు_tiled_calc_task(octx, grctx, i, n_k_tiles, tile_కొలత);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        compute_tiled(dst_spad, src_spad, n_k_tiles, tile_stride, task.అడ్డువరుస);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        // Real writeback of dst_spad
        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(task.dst_దత్తాంశం, dst_spad),
                       dst_బైట్లు, vtcm_layout->dst_spad_half_కొలత, dst_బైట్లు, 1);

        స్థిరరాశి uint32_t తదుపరి_అడుగు = అడుగు + 2;
        ఒకవేళ (తదుపరి_అడుగు < ir1 - ir0) {
            స్థిరరాశి uint32_t ni = ir0 + తదుపరి_అడుగు;
            నిర్మాణం get_అడ్డువరుసలు_tiled_task తదుపరి_task = get_అడ్డువరుసలు_tiled_calc_task(octx, grctx, ni, n_k_tiles, tile_కొలత);
            dma_వరుస_నెట్టు(dma_q,
                           dma_make_దత్తాంశం(src_spad, తదుపరి_task.tile_src_పునాది),
                           tile_stride, tile_కొలత, tile_కొలత, n_k_tiles);
        }
    }

    dma_వరుస_flush(dma_q);
}

స్థిర __attribute__((noinline)) శూన్య get_అడ్డువరుసలు_dma_src1(
    నిర్మాణం htp_ops_context * octx,
    uint8_t * vtcm_src1,
    uint32_t elem_కొలత
) {
    స్థిరరాశి uint32_t ne10 = octx->src[1]->ne[0];
    స్థిరరాశి uint32_t ne11 = octx->src[1]->ne[1];
    స్థిరరాశి uint32_t ne12 = octx->src[1]->ne[2];
    స్థిరరాశి uint32_t ne13 = octx->src[1]->ne[3];
    స్థిరరాశి uint32_t nb11 = octx->src[1]->nb[1];
    స్థిరరాశి uint32_t nb12 = octx->src[1]->nb[2];
    స్థిరరాశి uint32_t nb13 = octx->src[1]->nb[3];

    dma_వరుస * dma_q = octx->ctx->dma[0];
    uint8_t * cur_dst = vtcm_src1;
    కొరకు (uint32_t i13 = 0; i13 < ne13; ++i13) {
        కొరకు (uint32_t i12 = 0; i12 < ne12; ++i12) {
            స్థిరరాశి dma_addr_t src_plane = octx->src[1]->దత్తాంశం + i12 * nb12 + i13 * nb13;
            అయితే (!dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(cur_dst, src_plane),
                                   ne10 * elem_కొలత, nb11, ne10 * elem_కొలత, ne11)) {
                dma_వరుస_తీయి(dma_q);
            }
            cur_dst += ne10 * ne11 * elem_కొలత;
        }
    }
    dma_వరుస_flush(dma_q);
}

పూర్ణాంకం op_get_అడ్డువరుసలు(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params *) octx->kernel_params;

    స్థిరరాశి uint32_t ne00          = octx->src[0]->ne[0];
    స్థిరరాశి uint32_t src0_అడ్డువరుస_కొలత = htp_tensor_get_అడ్డువరుస_కొలత(octx->src[0]->type, ne00);
    ఒకవేళ (src0_అడ్డువరుస_కొలత == 0) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (kparams->kernel_type == HTP_GET_అడ్డువరుసలు_KERNEL_SAMETYPE) {
        ఒకవేళ (octx->src[0]->type != octx->dst->type) {
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
    } లేకపోతే {
        ఒకవేళ (octx->dst->type != HTP_TYPE_F32) {
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
    }

    ఒకవేళ (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    స్థిరరాశి uint32_t మొత్తం_tasks    = kparams->మొత్తం_tasks;
    స్థిరరాశి uint32_t dst_అడ్డువరుస_కొలత   = (kparams->kernel_type == HTP_GET_అడ్డువరుసలు_KERNEL_SAMETYPE) ? src0_అడ్డువరుస_కొలత : ne00 * పరిమాణం(దశాంశం);

    uint32_t task_ప్రారంభం = 0;
    uint32_t tasks      = మొత్తం_tasks;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t tasks_చొప్పున_chunk = 1;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, dst_అడ్డువరుస_కొలత / dst->ne[0], dst_అడ్డువరుస_కొలత, &tasks_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_tasks, tasks_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        task_ప్రారంభం = range.ప్రారంభం;
        tasks      = range.లెక్క;
    }

    ఒకవేళ (tasks == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, (uint32_t) kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    నిర్మాణం get_అడ్డువరుసలు_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_పునాది = (uint8_t *)octx->ctx->vtcm_పునాది;
    grctx.compute_flat = ఖాళీ;
    grctx.compute_tiled = ఖాళీ;
    grctx.task_ప్రారంభం = task_ప్రారంభం;
    grctx.tasks = tasks;
    grctx.tasks_చొప్పున_thread = octx->ctx->mdev.లెక్క == 1 ? kparams->tasks_చొప్పున_thread : fastdiv(tasks + n_దారాలు - 1, &octx->n_దారాలు_div);
    grctx.tile_కొలత   = 0;
    grctx.tile_stride = 0;
    grctx.src0_అడ్డువరుస_కొలత = src0_అడ్డువరుస_కొలత;
    grctx.dst_అడ్డువరుస_కొలత  = dst_అడ్డువరుస_కొలత;
    grctx.idx_type      = octx->src[1]->type;

    htp_get_అడ్డువరుసలు_vtcm_layout_build(&grctx.vtcm_layout, kparams->kernel_type, octx->src[0]->type, ne00, మొత్తం_tasks, grctx.idx_type, n_దారాలు);

    ఒకవేళ (grctx.vtcm_layout.మొత్తం_బైట్లు > octx->ctx->vtcm_కొలత) {
        FARF(ERROR, "get-rows: VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_కొలత, grctx.vtcm_layout.మొత్తం_బైట్లు);
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    work_వరుస_func_t q_func = ఖాళీ;
    ఎంపిక (kparams->kernel_type) {
        స్థితి HTP_GET_అడ్డువరుసలు_KERNEL_SAMETYPE:
            q_func = get_అడ్డువరుసలు_thread_sametype;
            ఆపు;
        స్థితి HTP_GET_అడ్డువరుసలు_KERNEL_TILED:
            ఎంపిక (octx->src[0]->type) {
                స్థితి HTP_TYPE_F16:  grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_f16_f32;  ఆపు;
                స్థితి HTP_TYPE_Q1_0: grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_q1_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q2_0: grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_q2_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q8_0: grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_q8_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q4_0: grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_q4_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q4_K: grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_q4_k_f32; ఆపు;
                స్థితి HTP_TYPE_Q5_K: grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_q5_k_f32; ఆపు;
                స్థితి HTP_TYPE_Q6_K: grctx.compute_tiled = hvx_dequantize_అడ్డువరుస_tiled_q6_k_f32; ఆపు;
                అప్రమేయ:            తిరిగి HTP_స్థితి_NO_SUPPORT;
            }
            grctx.tile_కొలత   = htp_tensor_get_tile_కొలత(octx->src[0]->type);
            grctx.tile_stride = htp_tensor_get_tile_padded_కొలత(octx->src[0]->type);
            q_func = get_అడ్డువరుసలు_thread_tiled;
            ఆపు;
        స్థితి HTP_GET_అడ్డువరుసలు_KERNEL_FLAT:
            ఎంపిక (octx->src[0]->type) {
                స్థితి HTP_TYPE_F16:  grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_f16_f32;  ఆపు;
                స్థితి HTP_TYPE_BF16: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_bf16_f32; ఆపు;
                స్థితి HTP_TYPE_Q1_0: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_q1_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q2_0: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_q2_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q8_0: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_q8_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q4_0: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_q4_0_f32; ఆపు;
                స్థితి HTP_TYPE_Q4_K: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_q4_k_f32; ఆపు;
                స్థితి HTP_TYPE_Q5_K: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_q5_k_f32; ఆపు;
                స్థితి HTP_TYPE_Q6_K: grctx.compute_flat = hvx_dequantize_అడ్డువరుస_flat_q6_k_f32; ఆపు;
                అప్రమేయ:            తిరిగి HTP_స్థితి_NO_SUPPORT;
            }
            q_func = get_అడ్డువరుసలు_thread_flat;
            ఆపు;
        అప్రమేయ:
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t elem_కొలత = (grctx.idx_type == HTP_TYPE_I64) ? పరిమాణం(int64_t) : పరిమాణం(int32_t);
    స్థిరరాశి size_t src1_బైట్లు  = (size_t) మొత్తం_tasks * elem_కొలత;

    ఒకవేళ (!htp_tensor_is_extended(octx->src[1]) && src1_బైట్లు <= 128 &&
        htp_tensor_is_contiguous(octx->src[1], elem_కొలత)) {
        grctx.src1_దత్తాంశం  = (స్థిరరాశి శూన్య *) (uintptr_t) octx->src[1]->దత్తాంశం;
        grctx.read_సూచీ = get_అడ్డువరుసలు_read_సూచీ_ddr;
    } లేకపోతే {
        uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, grctx.vtcm_పునాది, grctx.vtcm_layout.off_src1);
        get_అడ్డువరుసలు_dma_src1(octx, vtcm_src1, elem_కొలత);
        grctx.src1_దత్తాంశం  = vtcm_src1;
        grctx.read_సూచీ = get_అడ్డువరుసలు_read_సూచీ_vtcm;
    }

    FARF(ఎగువ_2, "get-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu kernel-type %d n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         grctx.vtcm_layout.src0_బైట్లు_చొప్పున_thread * n_దారాలు,
         grctx.vtcm_layout.dst_బైట్లు_చొప్పున_thread  * n_దారాలు,
         kparams->kernel_type, n_దారాలు);

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, q_func, &grctx, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}
