#شامل "gather-rows.h"

#شامل <assert.h>

#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "hvx-utils.h"
#شامل "dma-queue.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "work-queue.h"

قسمتعریف خالی (*gather_صفیں_compute_fn_t)(
    uint8_t * محدود برآمد, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت ساخت htp_gather_صفیں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur);

ساخت gather_صفیں_task {
    ساخت htp_ops_context *              octx;
    ثابت ساخت htp_tensor *             a;
    ثابت ساخت htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_آغاز;
    uint32_t                              صفیں;
    uint32_t                              صفیں_فی_thread;
    ثابت ساخت htp_gather_صفیں_params * kparams;
    gather_صفیں_compute_fn_t              compute;
};

ساکن اندرونی __attribute__((always_inline)) خالی gather_صفیں_dma_اندر(
    dma_قطار * q, ثابت ساخت gather_صفیں_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_سائز) {
    ثابت ساخت htp_tensor * a = task->a;
    ثابت ساخت htp_tensor * b = task->b;
    ثابت ساخت htp_gather_صفیں_params * kparams = task->kparams;

    ثابت size_t a_سائز = (nr - 1) * a->nb[1] + kparams->na * elem_سائز;
    dma_قطار_ڈالو(q, dma_make_ڈیٹا(region, a->ڈیٹا + r * a->nb[1]), a_سائز, a_سائز, a_سائز, 1);
    اگر (b) {
        اگر (kparams->b_dense) {
            ثابت size_t b_سائز = (nr - 1) * b->nb[1] + kparams->nb * elem_سائز;
            dma_قطار_ڈالو(q, dma_make_ڈیٹا(region + kparams->v1, b->ڈیٹا + r * b->nb[1]), b_سائز, b_سائز, b_سائز, 1);
        } ورنہ {
            ثابت size_t b_سائز = (nr - 1) * b->nb[1] + elem_سائز;
            برائے (uint32_t k = 0; k < kparams->nb; k++) {
                dma_قطار_ڈالو(q, dma_make_ڈیٹا(region + kparams->v1 + k * kparams->span1,
                                                b->ڈیٹا + k * b->nb[0] + r * b->nb[1]), b_سائز, b_سائز, b_سائز, 1);
            }
        }
    }
}

ساکن اندرونی خالی gather_صفیں_dma_برآمد(dma_قطار * q, ثابت ساخت gather_صفیں_task * task,
                                       uint8_t * برآمد, uint32_t r, uint32_t nr, uint32_t elem_سائز) {
    ثابت ساخت htp_gather_صفیں_params * kparams = task->kparams;
    ثابت size_t برآمد_سائز = nr * kparams->ne * elem_سائز;
    dma_قطار_ڈالو(q, dma_make_ڈیٹا(task->dst + r * kparams->ne * elem_سائز, برآمد), برآمد_سائز, برآمد_سائز, برآمد_سائز, 1);
}

ساکن __attribute__((noinline)) خالی gather_صفیں_init_tables_f32(
    HVX_سمتیہ * tab, HVX_سمتیہ * inc,
    ثابت ساخت htp_gather_صفیں_params * kparams,
    ثابت ساخت htp_tensor * a, ثابت ساخت htp_tensor * b) {

    ثابت int32_t s_a = kparams->rg * a->nb[1];
    ثابت int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t صف = 0;
    uint32_t ستون = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    برائے (uint32_t k = 0; k < kparams->vg; k++) {
        برائے (uint32_t lane = 0; lane < 32; lane++) {
            اگر (ستون < kparams->na) {
                t_buf[lane] = صف * a->nb[1] + ستون * 4;
                s_buf[lane] = s_a;
            } ورنہ {
                t_buf[lane] = kparams->v1 + (ستون - kparams->na) * kparams->span1 + صف * b_nb1;
                s_buf[lane] = s_b;
            }
            اگر (++ستون == kparams->ne) {
                ستون = 0;
                صف++;
            }
        }
        tab[k] = *(ثابت HVX_سمتیہ *) t_buf;
        inc[k] = *(ثابت HVX_سمتیہ *) s_buf;
    }
}

ساکن __attribute__((noinline)) خالی gather_صفیں_init_tables_f16(
    HVX_سمتیہ * tab, HVX_سمتیہ * inc,
    ثابت ساخت htp_gather_صفیں_params * kparams,
    ثابت ساخت htp_tensor * a, ثابت ساخت htp_tensor * b) {

    ثابت int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ثابت int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t صف = 0;
    uint32_t ستون = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    برائے (uint32_t k = 0; k < kparams->vg; k++) {
        برائے (uint32_t lane = 0; lane < 64; lane++) {
            اگر (ستون < kparams->na) {
                t_buf[lane] = (int16_t) (صف * a->nb[1] + ستون * 2);
                s_buf[lane] = s_a;
            } ورنہ {
                t_buf[lane] = (int16_t) (kparams->v1 + (ستون - kparams->na) * kparams->span1 + صف * b_nb1);
                s_buf[lane] = s_b;
            }
            اگر (++ستون == kparams->ne) {
                ستون = 0;
                صف++;
            }
        }
        tab[k] = *(ثابت HVX_سمتیہ *) t_buf;
        inc[k] = *(ثابت HVX_سمتیہ *) s_buf;
    }
}

ساکن خالی gather_صفیں_compute_f32_vg1(
    uint8_t * محدود برآمد, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت ساخت htp_gather_صفیں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    (خالی) cur;
    ثابت uint32_t region_سائز = kparams->region_سائز;
    HVX_سمتیہ vcur = tab[0];
    ثابت HVX_سمتیہ vinc = inc[0];
    #ہدایت unroll(4)
    برائے (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_سمتیہ *) (برآمد + v * 128), (size_t) region, region_سائز, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

ساکن خالی gather_صفیں_compute_f32_vgn(
    uint8_t * محدود برآمد, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت ساخت htp_gather_صفیں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_سائز = kparams->region_سائز;
    برائے (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ہدایت unroll(4)
    برائے (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_سمتیہ *) (برآمد + v * 128), (size_t) region, region_سائز, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        اگر (++k == vg) {
            k = 0;
        }
    }
}

ساکن خالی gather_صفیں_compute_f16_vg1(
    uint8_t * محدود برآمد, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت ساخت htp_gather_صفیں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    (خالی) cur;
    ثابت uint32_t region_سائز = kparams->region_سائز;
    HVX_سمتیہ vcur = tab[0];
    ثابت HVX_سمتیہ vinc = inc[0];
    #ہدایت unroll(4)
    برائے (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_سمتیہ *) (برآمد + v * 128), (size_t) region, region_سائز, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

ساکن خالی gather_صفیں_compute_f16_vgn(
    uint8_t * محدود برآمد, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت ساخت htp_gather_صفیں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_سائز = kparams->region_سائز;
    برائے (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ہدایت unroll(4)
    برائے (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_سمتیہ *) (برآمد + v * 128), (size_t) region, region_سائز, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        اگر (++k == vg) {
            k = 0;
        }
    }
}

ساکن خالی gather_صفیں_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ثابت ساخت gather_صفیں_task * task = (ثابت ساخت gather_صفیں_task *) ڈیٹا;
    ثابت ساخت htp_gather_صفیں_params * kparams = task->kparams;
    ثابت ساخت htp_tensor * a = task->a;
    ثابت ساخت htp_tensor * b = task->b;

    ثابت uint32_t r_انحراف = MIN(ith * task->صفیں_فی_thread, task->صفیں);
    ثابت uint32_t r_beg    = task->r_آغاز + r_انحراف;
    ثابت uint32_t r_انتہا    = task->r_آغاز + MIN(r_انحراف + task->صفیں_فی_thread, task->صفیں);
    اگر (r_beg >= r_انتہا) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * بنیاد   = task->octx->ctx->vtcm_بنیاد + ith * kparams->قاش_2;
    HVX_سمتیہ * tab = (HVX_سمتیہ *) بنیاد;
    HVX_سمتیہ * inc = tab + kparams->vg;
    HVX_سمتیہ * cur = inc + kparams->vg;
    uint8_t * region_بنیاد = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_بنیاد, region_بنیاد + kparams->buf_stride };
    uint8_t * برآمد[2]    = { region[0] + kparams->برآمد_off, region[1] + kparams->برآمد_off };

    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_صفیں_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_قطار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_اندر   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0    = MIN(nr_max, r_انتہا - r_beg);

    dma_قطار_ڈالو(q, dma_make_ڈیٹا(task->dst, برآمد[0]), 0, 0, 0, 0); // dummy out
    gather_صفیں_dma_اندر(q, task, region[0], r_beg, nr0, 4);

    ثابت uint32_t r1 = r_beg + nr_max;
    اگر (r1 < r_انتہا) {
        ثابت uint32_t nr1 = MIN(nr_max, r_انتہا - r1);
        dma_قطار_ڈالو(q, dma_make_ڈیٹا(task->dst, برآمد[1]), 0, 0, 0, 0); // dummy out
        gather_صفیں_dma_اندر(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    برائے (uint32_t r = r_beg; r < r_انتہا; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_انتہا - r);
        ثابت uint32_t اگلا_buf = buf ^ 1;

        dma_قطار_نکالو(q); // complete out
        برائے (uint32_t i = 0; i < n_اندر; i++) {
            dma_قطار_نکالو(q);
        }

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(برآمد[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(برآمد[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_صفیں_dma_برآمد(q, task, برآمد[buf], r, nr, 4);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        اگر (r_prefetch < r_انتہا) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_انتہا - r_prefetch);
            gather_صفیں_dma_اندر(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = اگلا_buf;
    }
    dma_قطار_flush(q);
}

ساکن خالی gather_صفیں_thread_f16(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ثابت ساخت gather_صفیں_task * task = (ثابت ساخت gather_صفیں_task *) ڈیٹا;
    ثابت ساخت htp_gather_صفیں_params * kparams = task->kparams;
    ثابت ساخت htp_tensor * a = task->a;
    ثابت ساخت htp_tensor * b = task->b;

    ثابت uint32_t r_انحراف = MIN(ith * task->صفیں_فی_thread, task->صفیں);
    ثابت uint32_t r_beg    = task->r_آغاز + r_انحراف;
    ثابت uint32_t r_انتہا    = task->r_آغاز + MIN(r_انحراف + task->صفیں_فی_thread, task->صفیں);
    اگر (r_beg >= r_انتہا) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * بنیاد   = task->octx->ctx->vtcm_بنیاد + ith * kparams->قاش_2;
    HVX_سمتیہ * tab = (HVX_سمتیہ *) بنیاد;
    HVX_سمتیہ * inc = tab + kparams->vg;
    HVX_سمتیہ * cur = inc + kparams->vg;
    uint8_t * region_بنیاد = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_بنیاد, region_بنیاد + kparams->buf_stride };
    uint8_t * برآمد[2]    = { region[0] + kparams->برآمد_off, region[1] + kparams->برآمد_off };

    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_صفیں_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_قطار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_اندر = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0 = MIN(nr_max, r_انتہا - r_beg);
    dma_قطار_ڈالو(q, dma_make_ڈیٹا(task->dst, برآمد[0]), 0, 0, 0, 0); // dummy out
    gather_صفیں_dma_اندر(q, task, region[0], r_beg, nr0, 2);

    ثابت uint32_t r1 = r_beg + nr_max;
    اگر (r1 < r_انتہا) {
        ثابت uint32_t nr1 = MIN(nr_max, r_انتہا - r1);
        dma_قطار_ڈالو(q, dma_make_ڈیٹا(task->dst, برآمد[1]), 0, 0, 0, 0); // dummy out
        gather_صفیں_dma_اندر(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    برائے (uint32_t r = r_beg; r < r_انتہا; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_انتہا - r);
        ثابت uint32_t اگلا_buf = buf ^ 1;

        dma_قطار_نکالو(q); // complete out
        برائے (uint32_t i = 0; i < n_اندر; i++) {
            dma_قطار_نکالو(q);
        }

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(برآمد[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(برآمد[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_صفیں_dma_برآمد(q, task, برآمد[buf], r, nr, 2);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        اگر (r_prefetch < r_انتہا) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_انتہا - r_prefetch);
            gather_صفیں_dma_اندر(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = اگلا_buf;
    }
    dma_قطار_flush(q);
}

صحیح htp_gather_صفیں(
    ساخت htp_ops_context * octx,
    ثابت ساخت htp_tensor * a,
    ثابت ساخت htp_tensor * b,
    ثابت ساخت htp_gather_صفیں_params * kparams,
    uint32_t n_دھاگے) {

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    أكد(kparams->قاش_2 * octx->n_دھاگے <= octx->ctx->vtcm_سائز);

    ثابت uint32_t کل_صفیں = kparams->کل_صفیں;
    uint32_t r_آغاز = 0;
    uint32_t r_گنتی = کل_صفیں;
    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            کل_صفیں, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        r_آغاز = range.آغاز;
        r_گنتی = range.گنتی;
    }

    اگر (r_گنتی == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t elem_سائز = (a->type == HTP_TYPE_F16) ? 2 : 4;

    ساخت gather_صفیں_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->ڈیٹا;
    task.r_آغاز         = r_آغاز;
    task.صفیں            = r_گنتی;
    task.kparams         = kparams;
    task.صفیں_فی_thread = kparams->صفیں_فی_thread;

    اگر (elem_سائز == 2) {
        task.compute = (kparams->vg == 1) ? gather_صفیں_compute_f16_vg1 : gather_صفیں_compute_f16_vgn;
    } ورنہ {
        task.compute = (kparams->vg == 1) ? gather_صفیں_compute_f32_vg1 : gather_صفیں_compute_f32_vgn;
    }

    work_قطار_func_t کارکن_func = (elem_سائز == 2) ? gather_صفیں_thread_f16 : gather_صفیں_thread_f32;
    work_قطار_تسلسل(octx->ctx->work_قطار, کارکن_func, &task, octx->n_دھاگے);
    واپس HTP_حالت_OK;
}
