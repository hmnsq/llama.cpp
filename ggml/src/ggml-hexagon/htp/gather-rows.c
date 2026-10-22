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

تعريفنوع فراغ (*gather_صفاں_compute_fn_t)(
    uint8_t * محدود باہرلا, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفاں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur);

بنية gather_صفاں_task {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_tensor *             a;
    ثابت بنية htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_شروع;
    uint32_t                              صفاں;
    uint32_t                              صفاں_فی_thread;
    ثابت بنية htp_gather_صفاں_params * kparams;
    gather_صفاں_compute_fn_t              compute;
};

ساكن مضمن __attribute__((always_inline)) فراغ gather_صفاں_dma_وچ(
    dma_قطار * q, ثابت بنية gather_صفاں_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_ناپ) {
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;
    ثابت بنية htp_gather_صفاں_params * kparams = task->kparams;

    ثابت size_t a_ناپ = (nr - 1) * a->nb[1] + kparams->na * elem_ناپ;
    dma_قطار_ڈالو(q, dma_make_انکڑے(region, a->انکڑے + r * a->nb[1]), a_ناپ, a_ناپ, a_ناپ, 1);
    إذا (b) {
        إذا (kparams->b_dense) {
            ثابت size_t b_ناپ = (nr - 1) * b->nb[1] + kparams->nb * elem_ناپ;
            dma_قطار_ڈالو(q, dma_make_انکڑے(region + kparams->v1, b->انکڑے + r * b->nb[1]), b_ناپ, b_ناپ, b_ناپ, 1);
        } وإلا {
            ثابت size_t b_ناپ = (nr - 1) * b->nb[1] + elem_ناپ;
            لكل (uint32_t k = 0; k < kparams->nb; k++) {
                dma_قطار_ڈالو(q, dma_make_انکڑے(region + kparams->v1 + k * kparams->span1,
                                                b->انکڑے + k * b->nb[0] + r * b->nb[1]), b_ناپ, b_ناپ, b_ناپ, 1);
            }
        }
    }
}

ساكن مضمن فراغ gather_صفاں_dma_باہرلا(dma_قطار * q, ثابت بنية gather_صفاں_task * task,
                                       uint8_t * باہرلا, uint32_t r, uint32_t nr, uint32_t elem_ناپ) {
    ثابت بنية htp_gather_صفاں_params * kparams = task->kparams;
    ثابت size_t باہرلا_ناپ = nr * kparams->ne * elem_ناپ;
    dma_قطار_ڈالو(q, dma_make_انکڑے(task->dst + r * kparams->ne * elem_ناپ, باہرلا), باہرلا_ناپ, باہرلا_ناپ, باہرلا_ناپ, 1);
}

ساكن __attribute__((noinline)) فراغ gather_صفاں_init_tables_f32(
    HVX_سمتیہ * tab, HVX_سمتیہ * inc,
    ثابت بنية htp_gather_صفاں_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int32_t s_a = kparams->rg * a->nb[1];
    ثابت int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t صف = 0;
    uint32_t خانہ = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 32; lane++) {
            إذا (خانہ < kparams->na) {
                t_buf[lane] = صف * a->nb[1] + خانہ * 4;
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = kparams->v1 + (خانہ - kparams->na) * kparams->span1 + صف * b_nb1;
                s_buf[lane] = s_b;
            }
            إذا (++خانہ == kparams->ne) {
                خانہ = 0;
                صف++;
            }
        }
        tab[k] = *(ثابت HVX_سمتیہ *) t_buf;
        inc[k] = *(ثابت HVX_سمتیہ *) s_buf;
    }
}

ساكن __attribute__((noinline)) فراغ gather_صفاں_init_tables_f16(
    HVX_سمتیہ * tab, HVX_سمتیہ * inc,
    ثابت بنية htp_gather_صفاں_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ثابت int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t صف = 0;
    uint32_t خانہ = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 64; lane++) {
            إذا (خانہ < kparams->na) {
                t_buf[lane] = (int16_t) (صف * a->nb[1] + خانہ * 2);
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = (int16_t) (kparams->v1 + (خانہ - kparams->na) * kparams->span1 + صف * b_nb1);
                s_buf[lane] = s_b;
            }
            إذا (++خانہ == kparams->ne) {
                خانہ = 0;
                صف++;
            }
        }
        tab[k] = *(ثابت HVX_سمتیہ *) t_buf;
        inc[k] = *(ثابت HVX_سمتیہ *) s_buf;
    }
}

ساكن فراغ gather_صفاں_compute_f32_vg1(
    uint8_t * محدود باہرلا, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفاں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_ناپ = kparams->region_ناپ;
    HVX_سمتیہ vcur = tab[0];
    ثابت HVX_سمتیہ vinc = inc[0];
    #ہدایت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_سمتیہ *) (باہرلا + v * 128), (size_t) region, region_ناپ, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

ساكن فراغ gather_صفاں_compute_f32_vgn(
    uint8_t * محدود باہرلا, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفاں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_ناپ = kparams->region_ناپ;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ہدایت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_سمتیہ *) (باہرلا + v * 128), (size_t) region, region_ناپ, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_صفاں_compute_f16_vg1(
    uint8_t * محدود باہرلا, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفاں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_ناپ = kparams->region_ناپ;
    HVX_سمتیہ vcur = tab[0];
    ثابت HVX_سمتیہ vinc = inc[0];
    #ہدایت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_سمتیہ *) (باہرلا + v * 128), (size_t) region, region_ناپ, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

ساكن فراغ gather_صفاں_compute_f16_vgn(
    uint8_t * محدود باہرلا, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفاں_params * محدود kparams,
    ثابت HVX_سمتیہ * محدود tab, ثابت HVX_سمتیہ * محدود inc, HVX_سمتیہ * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_ناپ = kparams->region_ناپ;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ہدایت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_سمتیہ *) (باہرلا + v * 128), (size_t) region, region_ناپ, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_صفاں_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    (فراغ) nth;
    ثابت بنية gather_صفاں_task * task = (ثابت بنية gather_صفاں_task *) انکڑے;
    ثابت بنية htp_gather_صفاں_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_وتھ = MIN(ith * task->صفاں_فی_thread, task->صفاں);
    ثابت uint32_t r_beg    = task->r_شروع + r_وتھ;
    ثابت uint32_t r_سرا    = task->r_شروع + MIN(r_وتھ + task->صفاں_فی_thread, task->صفاں);
    إذا (r_beg >= r_سرا) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * ادھار   = task->octx->ctx->vtcm_ادھار + ith * kparams->ٹکڑا_2;
    HVX_سمتیہ * tab = (HVX_سمتیہ *) ادھار;
    HVX_سمتیہ * inc = tab + kparams->vg;
    HVX_سمتیہ * cur = inc + kparams->vg;
    uint8_t * region_ادھار = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_ادھار, region_ادھار + kparams->buf_stride };
    uint8_t * باہرلا[2]    = { region[0] + kparams->باہرلا_off, region[1] + kparams->باہرلا_off };

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_صفاں_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_قطار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_وچ   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0    = MIN(nr_max, r_سرا - r_beg);

    dma_قطار_ڈالو(q, dma_make_انکڑے(task->dst, باہرلا[0]), 0, 0, 0, 0); // dummy out
    gather_صفاں_dma_وچ(q, task, region[0], r_beg, nr0, 4);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_سرا) {
        ثابت uint32_t nr1 = MIN(nr_max, r_سرا - r1);
        dma_قطار_ڈالو(q, dma_make_انکڑے(task->dst, باہرلا[1]), 0, 0, 0, 0); // dummy out
        gather_صفاں_dma_وچ(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_سرا; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_سرا - r);
        ثابت uint32_t اگلا_buf = buf ^ 1;

        dma_قطار_نکالو(q); // complete out
        لكل (uint32_t i = 0; i < n_وچ; i++) {
            dma_قطار_نکالو(q);
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(باہرلا[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(باہرلا[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_صفاں_dma_باہرلا(q, task, باہرلا[buf], r, nr, 4);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_سرا) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_سرا - r_prefetch);
            gather_صفاں_dma_وچ(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = اگلا_buf;
    }
    dma_قطار_flush(q);
}

ساكن فراغ gather_صفاں_thread_f16(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    (فراغ) nth;
    ثابت بنية gather_صفاں_task * task = (ثابت بنية gather_صفاں_task *) انکڑے;
    ثابت بنية htp_gather_صفاں_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_وتھ = MIN(ith * task->صفاں_فی_thread, task->صفاں);
    ثابت uint32_t r_beg    = task->r_شروع + r_وتھ;
    ثابت uint32_t r_سرا    = task->r_شروع + MIN(r_وتھ + task->صفاں_فی_thread, task->صفاں);
    إذا (r_beg >= r_سرا) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * ادھار   = task->octx->ctx->vtcm_ادھار + ith * kparams->ٹکڑا_2;
    HVX_سمتیہ * tab = (HVX_سمتیہ *) ادھار;
    HVX_سمتیہ * inc = tab + kparams->vg;
    HVX_سمتیہ * cur = inc + kparams->vg;
    uint8_t * region_ادھار = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_ادھار, region_ادھار + kparams->buf_stride };
    uint8_t * باہرلا[2]    = { region[0] + kparams->باہرلا_off, region[1] + kparams->باہرلا_off };

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_صفاں_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_قطار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_وچ = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0 = MIN(nr_max, r_سرا - r_beg);
    dma_قطار_ڈالو(q, dma_make_انکڑے(task->dst, باہرلا[0]), 0, 0, 0, 0); // dummy out
    gather_صفاں_dma_وچ(q, task, region[0], r_beg, nr0, 2);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_سرا) {
        ثابت uint32_t nr1 = MIN(nr_max, r_سرا - r1);
        dma_قطار_ڈالو(q, dma_make_انکڑے(task->dst, باہرلا[1]), 0, 0, 0, 0); // dummy out
        gather_صفاں_dma_وچ(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_سرا; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_سرا - r);
        ثابت uint32_t اگلا_buf = buf ^ 1;

        dma_قطار_نکالو(q); // complete out
        لكل (uint32_t i = 0; i < n_وچ; i++) {
            dma_قطار_نکالو(q);
        }

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(باہرلا[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(باہرلا[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_صفاں_dma_باہرلا(q, task, باہرلا[buf], r, nr, 2);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_سرا) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_سرا - r_prefetch);
            gather_صفاں_dma_وچ(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = اگلا_buf;
    }
    dma_قطار_flush(q);
}

صحيح htp_gather_صفاں(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_صفاں_params * kparams,
    uint32_t n_دھاگے) {

    إذا (!htp_ops_context_سیٹ_n_دھاگے(octx, n_دھاگے)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    أكد(kparams->ٹکڑا_2 * octx->n_دھاگے <= octx->ctx->vtcm_ناپ);

    ثابت uint32_t کل_صفاں = kparams->کل_صفاں;
    uint32_t r_شروع = 0;
    uint32_t r_گنتی = کل_صفاں;
    إذا (octx->ctx->mdev.گنتی > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(
            کل_صفاں, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        r_شروع = range.شروع;
        r_گنتی = range.گنتی;
    }

    إذا (r_گنتی == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t elem_ناپ = (a->type == HTP_TYPE_F16) ? 2 : 4;

    بنية gather_صفاں_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->انکڑے;
    task.r_شروع         = r_شروع;
    task.صفاں            = r_گنتی;
    task.kparams         = kparams;
    task.صفاں_فی_thread = kparams->صفاں_فی_thread;

    إذا (elem_ناپ == 2) {
        task.compute = (kparams->vg == 1) ? gather_صفاں_compute_f16_vg1 : gather_صفاں_compute_f16_vgn;
    } وإلا {
        task.compute = (kparams->vg == 1) ? gather_صفاں_compute_f32_vg1 : gather_صفاں_compute_f32_vgn;
    }

    work_قطار_func_t کامہ_func = (elem_ناپ == 2) ? gather_صفاں_thread_f16 : gather_صفاں_thread_f32;
    work_قطار_لگاتار(octx->ctx->work_قطار, کامہ_func, &task, octx->n_دھاگے);
    أرجع HTP_حالت_OK;
}
