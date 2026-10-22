#ضمن "gather-rows.h"

#ضمن <assert.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "hvx-utils.h"
#ضمن "dma-queue.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "work-queue.h"

تعريفنوع فراغ (*gather_ردیفها_compute_fn_t)(
    uint8_t * محدود بیرون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_ردیفها_params * محدود kparams,
    ثابت HVX_وکتور * محدود tab, ثابت HVX_وکتور * محدود inc, HVX_وکتور * محدود cur);

بنية gather_ردیفها_task {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_tensor *             a;
    ثابت بنية htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_شروع;
    uint32_t                              ردیفها;
    uint32_t                              ردیفها_در_هر_thread;
    ثابت بنية htp_gather_ردیفها_params * kparams;
    gather_ردیفها_compute_fn_t              compute;
};

ساكن مضمن __attribute__((always_inline)) فراغ gather_ردیفها_dma_درون(
    dma_صف * q, ثابت بنية gather_ردیفها_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_اندازه) {
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;
    ثابت بنية htp_gather_ردیفها_params * kparams = task->kparams;

    ثابت size_t a_اندازه = (nr - 1) * a->nb[1] + kparams->na * elem_اندازه;
    dma_صف_افزودن(q, dma_make_داده(region, a->داده + r * a->nb[1]), a_اندازه, a_اندازه, a_اندازه, 1);
    إذا (b) {
        إذا (kparams->b_dense) {
            ثابت size_t b_اندازه = (nr - 1) * b->nb[1] + kparams->nb * elem_اندازه;
            dma_صف_افزودن(q, dma_make_داده(region + kparams->v1, b->داده + r * b->nb[1]), b_اندازه, b_اندازه, b_اندازه, 1);
        } وإلا {
            ثابت size_t b_اندازه = (nr - 1) * b->nb[1] + elem_اندازه;
            لكل (uint32_t k = 0; k < kparams->nb; k++) {
                dma_صف_افزودن(q, dma_make_داده(region + kparams->v1 + k * kparams->span1,
                                                b->داده + k * b->nb[0] + r * b->nb[1]), b_اندازه, b_اندازه, b_اندازه, 1);
            }
        }
    }
}

ساكن مضمن فراغ gather_ردیفها_dma_بیرون(dma_صف * q, ثابت بنية gather_ردیفها_task * task,
                                       uint8_t * بیرون, uint32_t r, uint32_t nr, uint32_t elem_اندازه) {
    ثابت بنية htp_gather_ردیفها_params * kparams = task->kparams;
    ثابت size_t بیرون_اندازه = nr * kparams->ne * elem_اندازه;
    dma_صف_افزودن(q, dma_make_داده(task->dst + r * kparams->ne * elem_اندازه, بیرون), بیرون_اندازه, بیرون_اندازه, بیرون_اندازه, 1);
}

ساكن __attribute__((noinline)) فراغ gather_ردیفها_init_tables_f32(
    HVX_وکتور * tab, HVX_وکتور * inc,
    ثابت بنية htp_gather_ردیفها_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int32_t s_a = kparams->rg * a->nb[1];
    ثابت int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t ردیف = 0;
    uint32_t ستون = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 32; lane++) {
            إذا (ستون < kparams->na) {
                t_buf[lane] = ردیف * a->nb[1] + ستون * 4;
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = kparams->v1 + (ستون - kparams->na) * kparams->span1 + ردیف * b_nb1;
                s_buf[lane] = s_b;
            }
            إذا (++ستون == kparams->ne) {
                ستون = 0;
                ردیف++;
            }
        }
        tab[k] = *(ثابت HVX_وکتور *) t_buf;
        inc[k] = *(ثابت HVX_وکتور *) s_buf;
    }
}

ساكن __attribute__((noinline)) فراغ gather_ردیفها_init_tables_f16(
    HVX_وکتور * tab, HVX_وکتور * inc,
    ثابت بنية htp_gather_ردیفها_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ثابت int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t ردیف = 0;
    uint32_t ستون = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 64; lane++) {
            إذا (ستون < kparams->na) {
                t_buf[lane] = (int16_t) (ردیف * a->nb[1] + ستون * 2);
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = (int16_t) (kparams->v1 + (ستون - kparams->na) * kparams->span1 + ردیف * b_nb1);
                s_buf[lane] = s_b;
            }
            إذا (++ستون == kparams->ne) {
                ستون = 0;
                ردیف++;
            }
        }
        tab[k] = *(ثابت HVX_وکتور *) t_buf;
        inc[k] = *(ثابت HVX_وکتور *) s_buf;
    }
}

ساكن فراغ gather_ردیفها_compute_f32_vg1(
    uint8_t * محدود بیرون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_ردیفها_params * محدود kparams,
    ثابت HVX_وکتور * محدود tab, ثابت HVX_وکتور * محدود inc, HVX_وکتور * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    HVX_وکتور vcur = tab[0];
    ثابت HVX_وکتور vinc = inc[0];
    #رهنمود unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_وکتور *) (بیرون + v * 128), (size_t) region, region_اندازه, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

ساكن فراغ gather_ردیفها_compute_f32_vgn(
    uint8_t * محدود بیرون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_ردیفها_params * محدود kparams,
    ثابت HVX_وکتور * محدود tab, ثابت HVX_وکتور * محدود inc, HVX_وکتور * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #رهنمود unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_وکتور *) (بیرون + v * 128), (size_t) region, region_اندازه, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_ردیفها_compute_f16_vg1(
    uint8_t * محدود بیرون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_ردیفها_params * محدود kparams,
    ثابت HVX_وکتور * محدود tab, ثابت HVX_وکتور * محدود inc, HVX_وکتور * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    HVX_وکتور vcur = tab[0];
    ثابت HVX_وکتور vinc = inc[0];
    #رهنمود unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_وکتور *) (بیرون + v * 128), (size_t) region, region_اندازه, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

ساكن فراغ gather_ردیفها_compute_f16_vgn(
    uint8_t * محدود بیرون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_ردیفها_params * محدود kparams,
    ثابت HVX_وکتور * محدود tab, ثابت HVX_وکتور * محدود inc, HVX_وکتور * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #رهنمود unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_وکتور *) (بیرون + v * 128), (size_t) region, region_اندازه, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_ردیفها_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    ثابت بنية gather_ردیفها_task * task = (ثابت بنية gather_ردیفها_task *) داده;
    ثابت بنية htp_gather_ردیفها_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_جابه_جایی = MIN(ith * task->ردیفها_در_هر_thread, task->ردیفها);
    ثابت uint32_t r_beg    = task->r_شروع + r_جابه_جایی;
    ثابت uint32_t r_انتها    = task->r_شروع + MIN(r_جابه_جایی + task->ردیفها_در_هر_thread, task->ردیفها);
    إذا (r_beg >= r_انتها) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * پایه   = task->octx->ctx->vtcm_پایه + ith * kparams->برش_2;
    HVX_وکتور * tab = (HVX_وکتور *) پایه;
    HVX_وکتور * inc = tab + kparams->vg;
    HVX_وکتور * cur = inc + kparams->vg;
    uint8_t * region_پایه = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_پایه, region_پایه + kparams->buf_stride };
    uint8_t * بیرون[2]    = { region[0] + kparams->بیرون_off, region[1] + kparams->بیرون_off };

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_ردیفها_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_صف * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_درون   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0    = MIN(nr_max, r_انتها - r_beg);

    dma_صف_افزودن(q, dma_make_داده(task->dst, بیرون[0]), 0, 0, 0, 0); // dummy out
    gather_ردیفها_dma_درون(q, task, region[0], r_beg, nr0, 4);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_انتها) {
        ثابت uint32_t nr1 = MIN(nr_max, r_انتها - r1);
        dma_صف_افزودن(q, dma_make_داده(task->dst, بیرون[1]), 0, 0, 0, 0); // dummy out
        gather_ردیفها_dma_درون(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_انتها; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_انتها - r);
        ثابت uint32_t بعدی_buf = buf ^ 1;

        dma_صف_برداشت(q); // complete out
        لكل (uint32_t i = 0; i < n_درون; i++) {
            dma_صف_برداشت(q);
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(بیرون[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(بیرون[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_ردیفها_dma_بیرون(q, task, بیرون[buf], r, nr, 4);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_انتها) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_انتها - r_prefetch);
            gather_ردیفها_dma_درون(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = بعدی_buf;
    }
    dma_صف_flush(q);
}

ساكن فراغ gather_ردیفها_thread_f16(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    ثابت بنية gather_ردیفها_task * task = (ثابت بنية gather_ردیفها_task *) داده;
    ثابت بنية htp_gather_ردیفها_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_جابه_جایی = MIN(ith * task->ردیفها_در_هر_thread, task->ردیفها);
    ثابت uint32_t r_beg    = task->r_شروع + r_جابه_جایی;
    ثابت uint32_t r_انتها    = task->r_شروع + MIN(r_جابه_جایی + task->ردیفها_در_هر_thread, task->ردیفها);
    إذا (r_beg >= r_انتها) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * پایه   = task->octx->ctx->vtcm_پایه + ith * kparams->برش_2;
    HVX_وکتور * tab = (HVX_وکتور *) پایه;
    HVX_وکتور * inc = tab + kparams->vg;
    HVX_وکتور * cur = inc + kparams->vg;
    uint8_t * region_پایه = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_پایه, region_پایه + kparams->buf_stride };
    uint8_t * بیرون[2]    = { region[0] + kparams->بیرون_off, region[1] + kparams->بیرون_off };

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_ردیفها_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_صف * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_درون = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0 = MIN(nr_max, r_انتها - r_beg);
    dma_صف_افزودن(q, dma_make_داده(task->dst, بیرون[0]), 0, 0, 0, 0); // dummy out
    gather_ردیفها_dma_درون(q, task, region[0], r_beg, nr0, 2);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_انتها) {
        ثابت uint32_t nr1 = MIN(nr_max, r_انتها - r1);
        dma_صف_افزودن(q, dma_make_داده(task->dst, بیرون[1]), 0, 0, 0, 0); // dummy out
        gather_ردیفها_dma_درون(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_انتها; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_انتها - r);
        ثابت uint32_t بعدی_buf = buf ^ 1;

        dma_صف_برداشت(q); // complete out
        لكل (uint32_t i = 0; i < n_درون; i++) {
            dma_صف_برداشت(q);
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(بیرون[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(بیرون[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_ردیفها_dma_بیرون(q, task, بیرون[buf], r, nr, 2);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_انتها) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_انتها - r_prefetch);
            gather_ردیفها_dma_درون(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = بعدی_buf;
    }
    dma_صف_flush(q);
}

صحيح htp_gather_ردیفها(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_ردیفها_params * kparams,
    uint32_t n_نخها) {

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    أكد(kparams->برش_2 * octx->n_نخها <= octx->ctx->vtcm_اندازه);

    ثابت uint32_t مجموع_ردیفها = kparams->مجموع_ردیفها;
    uint32_t r_شروع = 0;
    uint32_t r_شمار = مجموع_ردیفها;
    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            مجموع_ردیفها, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        r_شروع = range.شروع;
        r_شمار = range.شمار;
    }

    إذا (r_شمار == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t elem_اندازه = (a->type == HTP_TYPE_F16) ? 2 : 4;

    بنية gather_ردیفها_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->داده;
    task.r_شروع         = r_شروع;
    task.ردیفها            = r_شمار;
    task.kparams         = kparams;
    task.ردیفها_در_هر_thread = kparams->ردیفها_در_هر_thread;

    إذا (elem_اندازه == 2) {
        task.compute = (kparams->vg == 1) ? gather_ردیفها_compute_f16_vg1 : gather_ردیفها_compute_f16_vgn;
    } وإلا {
        task.compute = (kparams->vg == 1) ? gather_ردیفها_compute_f32_vg1 : gather_ردیفها_compute_f32_vgn;
    }

    work_صف_func_t کارگر_func = (elem_اندازه == 2) ? gather_ردیفها_thread_f16 : gather_ردیفها_thread_f32;
    work_صف_پیاپی(octx->ctx->work_صف, کارگر_func, &task, octx->n_نخها);
    أرجع HTP_وضعیت_OK;
}
