#అన్తర్భావ "gather-rows.h"

#అన్తర్భావ <assert.h>

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "work-queue.h"

రకనిర్వచనం శూన్య (*gather_అడ్డువరుసలు_compute_fn_t)(
    uint8_t * పరిమిత బహిర్గతం, స్థిరరాశి uint8_t * పరిమిత region, uint32_t nvec,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * పరిమిత kparams,
    స్థిరరాశి HVX_సదిశ * పరిమిత tab, స్థిరరాశి HVX_సదిశ * పరిమిత inc, HVX_సదిశ * పరిమిత cur);

నిర్మాణం gather_అడ్డువరుసలు_task {
    నిర్మాణం htp_ops_context *              octx;
    స్థిరరాశి నిర్మాణం htp_tensor *             a;
    స్థిరరాశి నిర్మాణం htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_ప్రారంభం;
    uint32_t                              అడ్డువరుసలు;
    uint32_t                              అడ్డువరుసలు_చొప్పున_thread;
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams;
    gather_అడ్డువరుసలు_compute_fn_t              compute;
};

స్థిర అంతఃవరుస __attribute__((always_inline)) శూన్య gather_అడ్డువరుసలు_dma_లోపల(
    dma_వరుస * q, స్థిరరాశి నిర్మాణం gather_అడ్డువరుసలు_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_కొలత) {
    స్థిరరాశి నిర్మాణం htp_tensor * a = task->a;
    స్థిరరాశి నిర్మాణం htp_tensor * b = task->b;
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams = task->kparams;

    స్థిరరాశి size_t a_కొలత = (nr - 1) * a->nb[1] + kparams->na * elem_కొలత;
    dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(region, a->దత్తాంశం + r * a->nb[1]), a_కొలత, a_కొలత, a_కొలత, 1);
    ఒకవేళ (b) {
        ఒకవేళ (kparams->b_dense) {
            స్థిరరాశి size_t b_కొలత = (nr - 1) * b->nb[1] + kparams->nb * elem_కొలత;
            dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(region + kparams->v1, b->దత్తాంశం + r * b->nb[1]), b_కొలత, b_కొలత, b_కొలత, 1);
        } లేకపోతే {
            స్థిరరాశి size_t b_కొలత = (nr - 1) * b->nb[1] + elem_కొలత;
            కొరకు (uint32_t k = 0; k < kparams->nb; k++) {
                dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(region + kparams->v1 + k * kparams->span1,
                                                b->దత్తాంశం + k * b->nb[0] + r * b->nb[1]), b_కొలత, b_కొలత, b_కొలత, 1);
            }
        }
    }
}

స్థిర అంతఃవరుస శూన్య gather_అడ్డువరుసలు_dma_బహిర్గతం(dma_వరుస * q, స్థిరరాశి నిర్మాణం gather_అడ్డువరుసలు_task * task,
                                       uint8_t * బహిర్గతం, uint32_t r, uint32_t nr, uint32_t elem_కొలత) {
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams = task->kparams;
    స్థిరరాశి size_t బహిర్గతం_కొలత = nr * kparams->ne * elem_కొలత;
    dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(task->dst + r * kparams->ne * elem_కొలత, బహిర్గతం), బహిర్గతం_కొలత, బహిర్గతం_కొలత, బహిర్గతం_కొలత, 1);
}

స్థిర __attribute__((noinline)) శూన్య gather_అడ్డువరుసలు_init_tables_f32(
    HVX_సదిశ * tab, HVX_సదిశ * inc,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams,
    స్థిరరాశి నిర్మాణం htp_tensor * a, స్థిరరాశి నిర్మాణం htp_tensor * b) {

    స్థిరరాశి int32_t s_a = kparams->rg * a->nb[1];
    స్థిరరాశి int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    స్థిరరాశి uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t అడ్డువరుస = 0;
    uint32_t నిలువు = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    కొరకు (uint32_t k = 0; k < kparams->vg; k++) {
        కొరకు (uint32_t lane = 0; lane < 32; lane++) {
            ఒకవేళ (నిలువు < kparams->na) {
                t_buf[lane] = అడ్డువరుస * a->nb[1] + నిలువు * 4;
                s_buf[lane] = s_a;
            } లేకపోతే {
                t_buf[lane] = kparams->v1 + (నిలువు - kparams->na) * kparams->span1 + అడ్డువరుస * b_nb1;
                s_buf[lane] = s_b;
            }
            ఒకవేళ (++నిలువు == kparams->ne) {
                నిలువు = 0;
                అడ్డువరుస++;
            }
        }
        tab[k] = *(స్థిరరాశి HVX_సదిశ *) t_buf;
        inc[k] = *(స్థిరరాశి HVX_సదిశ *) s_buf;
    }
}

స్థిర __attribute__((noinline)) శూన్య gather_అడ్డువరుసలు_init_tables_f16(
    HVX_సదిశ * tab, HVX_సదిశ * inc,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams,
    స్థిరరాశి నిర్మాణం htp_tensor * a, స్థిరరాశి నిర్మాణం htp_tensor * b) {

    స్థిరరాశి int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    స్థిరరాశి int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    స్థిరరాశి uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t అడ్డువరుస = 0;
    uint32_t నిలువు = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    కొరకు (uint32_t k = 0; k < kparams->vg; k++) {
        కొరకు (uint32_t lane = 0; lane < 64; lane++) {
            ఒకవేళ (నిలువు < kparams->na) {
                t_buf[lane] = (int16_t) (అడ్డువరుస * a->nb[1] + నిలువు * 2);
                s_buf[lane] = s_a;
            } లేకపోతే {
                t_buf[lane] = (int16_t) (kparams->v1 + (నిలువు - kparams->na) * kparams->span1 + అడ్డువరుస * b_nb1);
                s_buf[lane] = s_b;
            }
            ఒకవేళ (++నిలువు == kparams->ne) {
                నిలువు = 0;
                అడ్డువరుస++;
            }
        }
        tab[k] = *(స్థిరరాశి HVX_సదిశ *) t_buf;
        inc[k] = *(స్థిరరాశి HVX_సదిశ *) s_buf;
    }
}

స్థిర శూన్య gather_అడ్డువరుసలు_compute_f32_vg1(
    uint8_t * పరిమిత బహిర్గతం, స్థిరరాశి uint8_t * పరిమిత region, uint32_t nvec,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * పరిమిత kparams,
    స్థిరరాశి HVX_సదిశ * పరిమిత tab, స్థిరరాశి HVX_సదిశ * పరిమిత inc, HVX_సదిశ * పరిమిత cur) {
    (శూన్య) cur;
    స్థిరరాశి uint32_t region_కొలత = kparams->region_కొలత;
    HVX_సదిశ vcur = tab[0];
    స్థిరరాశి HVX_సదిశ vinc = inc[0];
    #సూచన unroll(4)
    కొరకు (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_సదిశ *) (బహిర్గతం + v * 128), (size_t) region, region_కొలత, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

స్థిర శూన్య gather_అడ్డువరుసలు_compute_f32_vgn(
    uint8_t * పరిమిత బహిర్గతం, స్థిరరాశి uint8_t * పరిమిత region, uint32_t nvec,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * పరిమిత kparams,
    స్థిరరాశి HVX_సదిశ * పరిమిత tab, స్థిరరాశి HVX_సదిశ * పరిమిత inc, HVX_సదిశ * పరిమిత cur) {
    స్థిరరాశి uint32_t vg = kparams->vg;
    స్థిరరాశి uint32_t region_కొలత = kparams->region_కొలత;
    కొరకు (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #సూచన unroll(4)
    కొరకు (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_సదిశ *) (బహిర్గతం + v * 128), (size_t) region, region_కొలత, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        ఒకవేళ (++k == vg) {
            k = 0;
        }
    }
}

స్థిర శూన్య gather_అడ్డువరుసలు_compute_f16_vg1(
    uint8_t * పరిమిత బహిర్గతం, స్థిరరాశి uint8_t * పరిమిత region, uint32_t nvec,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * పరిమిత kparams,
    స్థిరరాశి HVX_సదిశ * పరిమిత tab, స్థిరరాశి HVX_సదిశ * పరిమిత inc, HVX_సదిశ * పరిమిత cur) {
    (శూన్య) cur;
    స్థిరరాశి uint32_t region_కొలత = kparams->region_కొలత;
    HVX_సదిశ vcur = tab[0];
    స్థిరరాశి HVX_సదిశ vinc = inc[0];
    #సూచన unroll(4)
    కొరకు (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_సదిశ *) (బహిర్గతం + v * 128), (size_t) region, region_కొలత, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

స్థిర శూన్య gather_అడ్డువరుసలు_compute_f16_vgn(
    uint8_t * పరిమిత బహిర్గతం, స్థిరరాశి uint8_t * పరిమిత region, uint32_t nvec,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * పరిమిత kparams,
    స్థిరరాశి HVX_సదిశ * పరిమిత tab, స్థిరరాశి HVX_సదిశ * పరిమిత inc, HVX_సదిశ * పరిమిత cur) {
    స్థిరరాశి uint32_t vg = kparams->vg;
    స్థిరరాశి uint32_t region_కొలత = kparams->region_కొలత;
    కొరకు (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #సూచన unroll(4)
    కొరకు (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_సదిశ *) (బహిర్గతం + v * 128), (size_t) region, region_కొలత, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        ఒకవేళ (++k == vg) {
            k = 0;
        }
    }
}

స్థిర శూన్య gather_అడ్డువరుసలు_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం gather_అడ్డువరుసలు_task * task = (స్థిరరాశి నిర్మాణం gather_అడ్డువరుసలు_task *) దత్తాంశం;
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams = task->kparams;
    స్థిరరాశి నిర్మాణం htp_tensor * a = task->a;
    స్థిరరాశి నిర్మాణం htp_tensor * b = task->b;

    స్థిరరాశి uint32_t r_స్థానభ్రంశం = MIN(ith * task->అడ్డువరుసలు_చొప్పున_thread, task->అడ్డువరుసలు);
    స్థిరరాశి uint32_t r_beg    = task->r_ప్రారంభం + r_స్థానభ్రంశం;
    స్థిరరాశి uint32_t r_చివర    = task->r_ప్రారంభం + MIN(r_స్థానభ్రంశం + task->అడ్డువరుసలు_చొప్పున_thread, task->అడ్డువరుసలు);
    ఒకవేళ (r_beg >= r_చివర) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * పునాది   = task->octx->ctx->vtcm_పునాది + ith * kparams->ముక్క_2;
    HVX_సదిశ * tab = (HVX_సదిశ *) పునాది;
    HVX_సదిశ * inc = tab + kparams->vg;
    HVX_సదిశ * cur = inc + kparams->vg;
    uint8_t * region_పునాది = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_పునాది, region_పునాది + kparams->buf_stride };
    uint8_t * బహిర్గతం[2]    = { region[0] + kparams->బహిర్గతం_off, region[1] + kparams->బహిర్గతం_off };

    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_అడ్డువరుసలు_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_వరుస * q = task->octx->ctx->dma[ith];
    స్థిరరాశి uint32_t n_లోపల   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    స్థిరరాశి uint32_t nr_max = kparams->nr_max;
    స్థిరరాశి uint32_t nr0    = MIN(nr_max, r_చివర - r_beg);

    dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(task->dst, బహిర్గతం[0]), 0, 0, 0, 0); // dummy out
    gather_అడ్డువరుసలు_dma_లోపల(q, task, region[0], r_beg, nr0, 4);

    స్థిరరాశి uint32_t r1 = r_beg + nr_max;
    ఒకవేళ (r1 < r_చివర) {
        స్థిరరాశి uint32_t nr1 = MIN(nr_max, r_చివర - r1);
        dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(task->dst, బహిర్గతం[1]), 0, 0, 0, 0); // dummy out
        gather_అడ్డువరుసలు_dma_లోపల(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    కొరకు (uint32_t r = r_beg; r < r_చివర; r += nr_max) {
        స్థిరరాశి uint32_t nr = MIN(nr_max, r_చివర - r);
        స్థిరరాశి uint32_t తదుపరి_buf = buf ^ 1;

        dma_వరుస_తీయి(q); // complete out
        కొరకు (uint32_t i = 0; i < n_లోపల; i++) {
            dma_వరుస_తీయి(q);
        }

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        స్థిరరాశి uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(బహిర్గతం[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(బహిర్గతం[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_అడ్డువరుసలు_dma_బహిర్గతం(q, task, బహిర్గతం[buf], r, nr, 4);

        స్థిరరాశి uint32_t r_prefetch = r + 2 * nr_max;
        ఒకవేళ (r_prefetch < r_చివర) {
            స్థిరరాశి uint32_t nr_prefetch = MIN(nr_max, r_చివర - r_prefetch);
            gather_అడ్డువరుసలు_dma_లోపల(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = తదుపరి_buf;
    }
    dma_వరుస_flush(q);
}

స్థిర శూన్య gather_అడ్డువరుసలు_thread_f16(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం gather_అడ్డువరుసలు_task * task = (స్థిరరాశి నిర్మాణం gather_అడ్డువరుసలు_task *) దత్తాంశం;
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams = task->kparams;
    స్థిరరాశి నిర్మాణం htp_tensor * a = task->a;
    స్థిరరాశి నిర్మాణం htp_tensor * b = task->b;

    స్థిరరాశి uint32_t r_స్థానభ్రంశం = MIN(ith * task->అడ్డువరుసలు_చొప్పున_thread, task->అడ్డువరుసలు);
    స్థిరరాశి uint32_t r_beg    = task->r_ప్రారంభం + r_స్థానభ్రంశం;
    స్థిరరాశి uint32_t r_చివర    = task->r_ప్రారంభం + MIN(r_స్థానభ్రంశం + task->అడ్డువరుసలు_చొప్పున_thread, task->అడ్డువరుసలు);
    ఒకవేళ (r_beg >= r_చివర) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * పునాది   = task->octx->ctx->vtcm_పునాది + ith * kparams->ముక్క_2;
    HVX_సదిశ * tab = (HVX_సదిశ *) పునాది;
    HVX_సదిశ * inc = tab + kparams->vg;
    HVX_సదిశ * cur = inc + kparams->vg;
    uint8_t * region_పునాది = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_పునాది, region_పునాది + kparams->buf_stride };
    uint8_t * బహిర్గతం[2]    = { region[0] + kparams->బహిర్గతం_off, region[1] + kparams->బహిర్గతం_off };

    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_అడ్డువరుసలు_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_వరుస * q = task->octx->ctx->dma[ith];
    స్థిరరాశి uint32_t n_లోపల = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    స్థిరరాశి uint32_t nr_max = kparams->nr_max;
    స్థిరరాశి uint32_t nr0 = MIN(nr_max, r_చివర - r_beg);
    dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(task->dst, బహిర్గతం[0]), 0, 0, 0, 0); // dummy out
    gather_అడ్డువరుసలు_dma_లోపల(q, task, region[0], r_beg, nr0, 2);

    స్థిరరాశి uint32_t r1 = r_beg + nr_max;
    ఒకవేళ (r1 < r_చివర) {
        స్థిరరాశి uint32_t nr1 = MIN(nr_max, r_చివర - r1);
        dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(task->dst, బహిర్గతం[1]), 0, 0, 0, 0); // dummy out
        gather_అడ్డువరుసలు_dma_లోపల(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    కొరకు (uint32_t r = r_beg; r < r_చివర; r += nr_max) {
        స్థిరరాశి uint32_t nr = MIN(nr_max, r_చివర - r);
        స్థిరరాశి uint32_t తదుపరి_buf = buf ^ 1;

        dma_వరుస_తీయి(q); // complete out
        కొరకు (uint32_t i = 0; i < n_లోపల; i++) {
            dma_వరుస_తీయి(q);
        }

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        స్థిరరాశి uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(బహిర్గతం[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(బహిర్గతం[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_అడ్డువరుసలు_dma_బహిర్గతం(q, task, బహిర్గతం[buf], r, nr, 2);

        స్థిరరాశి uint32_t r_prefetch = r + 2 * nr_max;
        ఒకవేళ (r_prefetch < r_చివర) {
            స్థిరరాశి uint32_t nr_prefetch = MIN(nr_max, r_చివర - r_prefetch);
            gather_అడ్డువరుసలు_dma_లోపల(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = తదుపరి_buf;
    }
    dma_వరుస_flush(q);
}

పూర్ణాంకం htp_gather_అడ్డువరుసలు(
    నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం htp_tensor * a,
    స్థిరరాశి నిర్మాణం htp_tensor * b,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams,
    uint32_t n_దారాలు) {

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    ప్రతిజ్ఞా(kparams->ముక్క_2 * octx->n_దారాలు <= octx->ctx->vtcm_కొలత);

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = kparams->మొత్తం_అడ్డువరుసలు;
    uint32_t r_ప్రారంభం = 0;
    uint32_t r_లెక్క = మొత్తం_అడ్డువరుసలు;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            మొత్తం_అడ్డువరుసలు, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        r_ప్రారంభం = range.ప్రారంభం;
        r_లెక్క = range.లెక్క;
    }

    ఒకవేళ (r_లెక్క == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t elem_కొలత = (a->type == HTP_TYPE_F16) ? 2 : 4;

    నిర్మాణం gather_అడ్డువరుసలు_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->దత్తాంశం;
    task.r_ప్రారంభం         = r_ప్రారంభం;
    task.అడ్డువరుసలు            = r_లెక్క;
    task.kparams         = kparams;
    task.అడ్డువరుసలు_చొప్పున_thread = kparams->అడ్డువరుసలు_చొప్పున_thread;

    ఒకవేళ (elem_కొలత == 2) {
        task.compute = (kparams->vg == 1) ? gather_అడ్డువరుసలు_compute_f16_vg1 : gather_అడ్డువరుసలు_compute_f16_vgn;
    } లేకపోతే {
        task.compute = (kparams->vg == 1) ? gather_అడ్డువరుసలు_compute_f32_vg1 : gather_అడ్డువరుసలు_compute_f32_vgn;
    }

    work_వరుస_func_t కార్మికుడు_func = (elem_కొలత == 2) ? gather_అడ్డువరుసలు_thread_f16 : gather_అడ్డువరుసలు_thread_f32;
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, కార్మికుడు_func, &task, octx->n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}
