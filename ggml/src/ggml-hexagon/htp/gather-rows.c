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

تعريفنوع فراغ (*gather_صفون_compute_fn_t)(
    uint8_t * محدود ٻاهريون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفون_params * محدود kparams,
    ثابت HVX_ويڪٽر * محدود tab, ثابت HVX_ويڪٽر * محدود inc, HVX_ويڪٽر * محدود cur);

بنية gather_صفون_task {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_tensor *             a;
    ثابت بنية htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_شروعات;
    uint32_t                              صفون;
    uint32_t                              صفون_في_thread;
    ثابت بنية htp_gather_صفون_params * kparams;
    gather_صفون_compute_fn_t              compute;
};

ساكن مضمن __attribute__((always_inline)) فراغ gather_صفون_dma_اندريون(
    dma_قطار * q, ثابت بنية gather_صفون_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_سائيز) {
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;
    ثابت بنية htp_gather_صفون_params * kparams = task->kparams;

    ثابت size_t a_سائيز = (nr - 1) * a->nb[1] + kparams->na * elem_سائيز;
    dma_قطار_ڌڪو(q, dma_make_معلومات(region, a->معلومات + r * a->nb[1]), a_سائيز, a_سائيز, a_سائيز, 1);
    إذا (b) {
        إذا (kparams->b_dense) {
            ثابت size_t b_سائيز = (nr - 1) * b->nb[1] + kparams->nb * elem_سائيز;
            dma_قطار_ڌڪو(q, dma_make_معلومات(region + kparams->v1, b->معلومات + r * b->nb[1]), b_سائيز, b_سائيز, b_سائيز, 1);
        } وإلا {
            ثابت size_t b_سائيز = (nr - 1) * b->nb[1] + elem_سائيز;
            لكل (uint32_t k = 0; k < kparams->nb; k++) {
                dma_قطار_ڌڪو(q, dma_make_معلومات(region + kparams->v1 + k * kparams->span1,
                                                b->معلومات + k * b->nb[0] + r * b->nb[1]), b_سائيز, b_سائيز, b_سائيز, 1);
            }
        }
    }
}

ساكن مضمن فراغ gather_صفون_dma_ٻاهريون(dma_قطار * q, ثابت بنية gather_صفون_task * task,
                                       uint8_t * ٻاهريون, uint32_t r, uint32_t nr, uint32_t elem_سائيز) {
    ثابت بنية htp_gather_صفون_params * kparams = task->kparams;
    ثابت size_t ٻاهريون_سائيز = nr * kparams->ne * elem_سائيز;
    dma_قطار_ڌڪو(q, dma_make_معلومات(task->dst + r * kparams->ne * elem_سائيز, ٻاهريون), ٻاهريون_سائيز, ٻاهريون_سائيز, ٻاهريون_سائيز, 1);
}

ساكن __attribute__((noinline)) فراغ gather_صفون_init_tables_f32(
    HVX_ويڪٽر * tab, HVX_ويڪٽر * inc,
    ثابت بنية htp_gather_صفون_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int32_t s_a = kparams->rg * a->nb[1];
    ثابت int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t صف = 0;
    uint32_t کانو = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 32; lane++) {
            إذا (کانو < kparams->na) {
                t_buf[lane] = صف * a->nb[1] + کانو * 4;
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = kparams->v1 + (کانو - kparams->na) * kparams->span1 + صف * b_nb1;
                s_buf[lane] = s_b;
            }
            إذا (++کانو == kparams->ne) {
                کانو = 0;
                صف++;
            }
        }
        tab[k] = *(ثابت HVX_ويڪٽر *) t_buf;
        inc[k] = *(ثابت HVX_ويڪٽر *) s_buf;
    }
}

ساكن __attribute__((noinline)) فراغ gather_صفون_init_tables_f16(
    HVX_ويڪٽر * tab, HVX_ويڪٽر * inc,
    ثابت بنية htp_gather_صفون_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ثابت int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t صف = 0;
    uint32_t کانو = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 64; lane++) {
            إذا (کانو < kparams->na) {
                t_buf[lane] = (int16_t) (صف * a->nb[1] + کانو * 2);
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = (int16_t) (kparams->v1 + (کانو - kparams->na) * kparams->span1 + صف * b_nb1);
                s_buf[lane] = s_b;
            }
            إذا (++کانو == kparams->ne) {
                کانو = 0;
                صف++;
            }
        }
        tab[k] = *(ثابت HVX_ويڪٽر *) t_buf;
        inc[k] = *(ثابت HVX_ويڪٽر *) s_buf;
    }
}

ساكن فراغ gather_صفون_compute_f32_vg1(
    uint8_t * محدود ٻاهريون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفون_params * محدود kparams,
    ثابت HVX_ويڪٽر * محدود tab, ثابت HVX_ويڪٽر * محدود inc, HVX_ويڪٽر * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_سائيز = kparams->region_سائيز;
    HVX_ويڪٽر vcur = tab[0];
    ثابت HVX_ويڪٽر vinc = inc[0];
    #هدايت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ويڪٽر *) (ٻاهريون + v * 128), (size_t) region, region_سائيز, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

ساكن فراغ gather_صفون_compute_f32_vgn(
    uint8_t * محدود ٻاهريون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفون_params * محدود kparams,
    ثابت HVX_ويڪٽر * محدود tab, ثابت HVX_ويڪٽر * محدود inc, HVX_ويڪٽر * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_سائيز = kparams->region_سائيز;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #هدايت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ويڪٽر *) (ٻاهريون + v * 128), (size_t) region, region_سائيز, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_صفون_compute_f16_vg1(
    uint8_t * محدود ٻاهريون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفون_params * محدود kparams,
    ثابت HVX_ويڪٽر * محدود tab, ثابت HVX_ويڪٽر * محدود inc, HVX_ويڪٽر * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_سائيز = kparams->region_سائيز;
    HVX_ويڪٽر vcur = tab[0];
    ثابت HVX_ويڪٽر vinc = inc[0];
    #هدايت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ويڪٽر *) (ٻاهريون + v * 128), (size_t) region, region_سائيز, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

ساكن فراغ gather_صفون_compute_f16_vgn(
    uint8_t * محدود ٻاهريون, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_صفون_params * محدود kparams,
    ثابت HVX_ويڪٽر * محدود tab, ثابت HVX_ويڪٽر * محدود inc, HVX_ويڪٽر * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_سائيز = kparams->region_سائيز;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #هدايت unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ويڪٽر *) (ٻاهريون + v * 128), (size_t) region, region_سائيز, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_صفون_thread_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية gather_صفون_task * task = (ثابت بنية gather_صفون_task *) معلومات;
    ثابت بنية htp_gather_صفون_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_وٿي = MIN(ith * task->صفون_في_thread, task->صفون);
    ثابت uint32_t r_beg    = task->r_شروعات + r_وٿي;
    ثابت uint32_t r_پڇاڙي    = task->r_شروعات + MIN(r_وٿي + task->صفون_في_thread, task->صفون);
    إذا (r_beg >= r_پڇاڙي) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * بنياد   = task->octx->ctx->vtcm_بنياد + ith * kparams->ٽڪرو_2;
    HVX_ويڪٽر * tab = (HVX_ويڪٽر *) بنياد;
    HVX_ويڪٽر * inc = tab + kparams->vg;
    HVX_ويڪٽر * cur = inc + kparams->vg;
    uint8_t * region_بنياد = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_بنياد, region_بنياد + kparams->buf_stride };
    uint8_t * ٻاهريون[2]    = { region[0] + kparams->ٻاهريون_off, region[1] + kparams->ٻاهريون_off };

    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_صفون_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_قطار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_اندريون   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0    = MIN(nr_max, r_پڇاڙي - r_beg);

    dma_قطار_ڌڪو(q, dma_make_معلومات(task->dst, ٻاهريون[0]), 0, 0, 0, 0); // dummy out
    gather_صفون_dma_اندريون(q, task, region[0], r_beg, nr0, 4);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_پڇاڙي) {
        ثابت uint32_t nr1 = MIN(nr_max, r_پڇاڙي - r1);
        dma_قطار_ڌڪو(q, dma_make_معلومات(task->dst, ٻاهريون[1]), 0, 0, 0, 0); // dummy out
        gather_صفون_dma_اندريون(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_پڇاڙي; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_پڇاڙي - r);
        ثابت uint32_t ايندڙ_buf = buf ^ 1;

        dma_قطار_ڪڍو(q); // complete out
        لكل (uint32_t i = 0; i < n_اندريون; i++) {
            dma_قطار_ڪڍو(q);
        }

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(ٻاهريون[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(ٻاهريون[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_صفون_dma_ٻاهريون(q, task, ٻاهريون[buf], r, nr, 4);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_پڇاڙي) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_پڇاڙي - r_prefetch);
            gather_صفون_dma_اندريون(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = ايندڙ_buf;
    }
    dma_قطار_flush(q);
}

ساكن فراغ gather_صفون_thread_f16(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية gather_صفون_task * task = (ثابت بنية gather_صفون_task *) معلومات;
    ثابت بنية htp_gather_صفون_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_وٿي = MIN(ith * task->صفون_في_thread, task->صفون);
    ثابت uint32_t r_beg    = task->r_شروعات + r_وٿي;
    ثابت uint32_t r_پڇاڙي    = task->r_شروعات + MIN(r_وٿي + task->صفون_في_thread, task->صفون);
    إذا (r_beg >= r_پڇاڙي) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * بنياد   = task->octx->ctx->vtcm_بنياد + ith * kparams->ٽڪرو_2;
    HVX_ويڪٽر * tab = (HVX_ويڪٽر *) بنياد;
    HVX_ويڪٽر * inc = tab + kparams->vg;
    HVX_ويڪٽر * cur = inc + kparams->vg;
    uint8_t * region_بنياد = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_بنياد, region_بنياد + kparams->buf_stride };
    uint8_t * ٻاهريون[2]    = { region[0] + kparams->ٻاهريون_off, region[1] + kparams->ٻاهريون_off };

    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_صفون_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_قطار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_اندريون = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0 = MIN(nr_max, r_پڇاڙي - r_beg);
    dma_قطار_ڌڪو(q, dma_make_معلومات(task->dst, ٻاهريون[0]), 0, 0, 0, 0); // dummy out
    gather_صفون_dma_اندريون(q, task, region[0], r_beg, nr0, 2);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_پڇاڙي) {
        ثابت uint32_t nr1 = MIN(nr_max, r_پڇاڙي - r1);
        dma_قطار_ڌڪو(q, dma_make_معلومات(task->dst, ٻاهريون[1]), 0, 0, 0, 0); // dummy out
        gather_صفون_dma_اندريون(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_پڇاڙي; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_پڇاڙي - r);
        ثابت uint32_t ايندڙ_buf = buf ^ 1;

        dma_قطار_ڪڍو(q); // complete out
        لكل (uint32_t i = 0; i < n_اندريون; i++) {
            dma_قطار_ڪڍو(q);
        }

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(ٻاهريون[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(ٻاهريون[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_صفون_dma_ٻاهريون(q, task, ٻاهريون[buf], r, nr, 2);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_پڇاڙي) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_پڇاڙي - r_prefetch);
            gather_صفون_dma_اندريون(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = ايندڙ_buf;
    }
    dma_قطار_flush(q);
}

صحيح htp_gather_صفون(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_صفون_params * kparams,
    uint32_t n_ڌاڳا) {

    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    أكد(kparams->ٽڪرو_2 * octx->n_ڌاڳا <= octx->ctx->vtcm_سائيز);

    ثابت uint32_t ڪل_صفون = kparams->ڪل_صفون;
    uint32_t r_شروعات = 0;
    uint32_t r_ڳڻپ = ڪل_صفون;
    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            ڪل_صفون, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        r_شروعات = range.شروعات;
        r_ڳڻپ = range.ڳڻپ;
    }

    إذا (r_ڳڻپ == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t elem_سائيز = (a->type == HTP_TYPE_F16) ? 2 : 4;

    بنية gather_صفون_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->معلومات;
    task.r_شروعات         = r_شروعات;
    task.صفون            = r_ڳڻپ;
    task.kparams         = kparams;
    task.صفون_في_thread = kparams->صفون_في_thread;

    إذا (elem_سائيز == 2) {
        task.compute = (kparams->vg == 1) ? gather_صفون_compute_f16_vg1 : gather_صفون_compute_f16_vgn;
    } وإلا {
        task.compute = (kparams->vg == 1) ? gather_صفون_compute_f32_vg1 : gather_صفون_compute_f32_vgn;
    }

    work_قطار_func_t ڪم_ڪندڙ_func = (elem_سائيز == 2) ? gather_صفون_thread_f16 : gather_صفون_thread_f32;
    work_قطار_لاڳيتو(octx->ctx->work_قطار, ڪم_ڪندڙ_func, &task, octx->n_ڌاڳا);
    أرجع HTP_حالت_OK;
}
