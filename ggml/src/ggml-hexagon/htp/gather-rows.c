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

تعريفنوع فراغ (*gather_لیکې_compute_fn_t)(
    uint8_t * محدود بهر, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_لیکې_params * محدود kparams,
    ثابت HVX_ویکټور * محدود tab, ثابت HVX_ویکټور * محدود inc, HVX_ویکټور * محدود cur);

بنية gather_لیکې_task {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_tensor *             a;
    ثابت بنية htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_پیل;
    uint32_t                              لیکې;
    uint32_t                              لیکې_فی_thread;
    ثابت بنية htp_gather_لیکې_params * kparams;
    gather_لیکې_compute_fn_t              compute;
};

ساكن مضمن __attribute__((always_inline)) فراغ gather_لیکې_dma_دننه(
    dma_کتار * q, ثابت بنية gather_لیکې_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_اندازه) {
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;
    ثابت بنية htp_gather_لیکې_params * kparams = task->kparams;

    ثابت size_t a_اندازه = (nr - 1) * a->nb[1] + kparams->na * elem_اندازه;
    dma_کتار_ټېل(q, dma_make_معلومات(region, a->معلومات + r * a->nb[1]), a_اندازه, a_اندازه, a_اندازه, 1);
    إذا (b) {
        إذا (kparams->b_dense) {
            ثابت size_t b_اندازه = (nr - 1) * b->nb[1] + kparams->nb * elem_اندازه;
            dma_کتار_ټېل(q, dma_make_معلومات(region + kparams->v1, b->معلومات + r * b->nb[1]), b_اندازه, b_اندازه, b_اندازه, 1);
        } وإلا {
            ثابت size_t b_اندازه = (nr - 1) * b->nb[1] + elem_اندازه;
            لكل (uint32_t k = 0; k < kparams->nb; k++) {
                dma_کتار_ټېل(q, dma_make_معلومات(region + kparams->v1 + k * kparams->span1,
                                                b->معلومات + k * b->nb[0] + r * b->nb[1]), b_اندازه, b_اندازه, b_اندازه, 1);
            }
        }
    }
}

ساكن مضمن فراغ gather_لیکې_dma_بهر(dma_کتار * q, ثابت بنية gather_لیکې_task * task,
                                       uint8_t * بهر, uint32_t r, uint32_t nr, uint32_t elem_اندازه) {
    ثابت بنية htp_gather_لیکې_params * kparams = task->kparams;
    ثابت size_t بهر_اندازه = nr * kparams->ne * elem_اندازه;
    dma_کتار_ټېل(q, dma_make_معلومات(task->dst + r * kparams->ne * elem_اندازه, بهر), بهر_اندازه, بهر_اندازه, بهر_اندازه, 1);
}

ساكن __attribute__((noinline)) فراغ gather_لیکې_init_tables_f32(
    HVX_ویکټور * tab, HVX_ویکټور * inc,
    ثابت بنية htp_gather_لیکې_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int32_t s_a = kparams->rg * a->nb[1];
    ثابت int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t لیکه = 0;
    uint32_t ستون = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 32; lane++) {
            إذا (ستون < kparams->na) {
                t_buf[lane] = لیکه * a->nb[1] + ستون * 4;
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = kparams->v1 + (ستون - kparams->na) * kparams->span1 + لیکه * b_nb1;
                s_buf[lane] = s_b;
            }
            إذا (++ستون == kparams->ne) {
                ستون = 0;
                لیکه++;
            }
        }
        tab[k] = *(ثابت HVX_ویکټور *) t_buf;
        inc[k] = *(ثابت HVX_ویکټور *) s_buf;
    }
}

ساكن __attribute__((noinline)) فراغ gather_لیکې_init_tables_f16(
    HVX_ویکټور * tab, HVX_ویکټور * inc,
    ثابت بنية htp_gather_لیکې_params * kparams,
    ثابت بنية htp_tensor * a, ثابت بنية htp_tensor * b) {

    ثابت int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ثابت int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ثابت uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t لیکه = 0;
    uint32_t ستون = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    لكل (uint32_t k = 0; k < kparams->vg; k++) {
        لكل (uint32_t lane = 0; lane < 64; lane++) {
            إذا (ستون < kparams->na) {
                t_buf[lane] = (int16_t) (لیکه * a->nb[1] + ستون * 2);
                s_buf[lane] = s_a;
            } وإلا {
                t_buf[lane] = (int16_t) (kparams->v1 + (ستون - kparams->na) * kparams->span1 + لیکه * b_nb1);
                s_buf[lane] = s_b;
            }
            إذا (++ستون == kparams->ne) {
                ستون = 0;
                لیکه++;
            }
        }
        tab[k] = *(ثابت HVX_ویکټور *) t_buf;
        inc[k] = *(ثابت HVX_ویکټور *) s_buf;
    }
}

ساكن فراغ gather_لیکې_compute_f32_vg1(
    uint8_t * محدود بهر, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_لیکې_params * محدود kparams,
    ثابت HVX_ویکټور * محدود tab, ثابت HVX_ویکټور * محدود inc, HVX_ویکټور * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    HVX_ویکټور vcur = tab[0];
    ثابت HVX_ویکټور vinc = inc[0];
    #لارښوونه unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ویکټور *) (بهر + v * 128), (size_t) region, region_اندازه, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

ساكن فراغ gather_لیکې_compute_f32_vgn(
    uint8_t * محدود بهر, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_لیکې_params * محدود kparams,
    ثابت HVX_ویکټور * محدود tab, ثابت HVX_ویکټور * محدود inc, HVX_ویکټور * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #لارښوونه unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ویکټور *) (بهر + v * 128), (size_t) region, region_اندازه, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_لیکې_compute_f16_vg1(
    uint8_t * محدود بهر, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_لیکې_params * محدود kparams,
    ثابت HVX_ویکټور * محدود tab, ثابت HVX_ویکټور * محدود inc, HVX_ویکټور * محدود cur) {
    (فراغ) cur;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    HVX_ویکټور vcur = tab[0];
    ثابت HVX_ویکټور vinc = inc[0];
    #لارښوونه unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ویکټور *) (بهر + v * 128), (size_t) region, region_اندازه, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

ساكن فراغ gather_لیکې_compute_f16_vgn(
    uint8_t * محدود بهر, ثابت uint8_t * محدود region, uint32_t nvec,
    ثابت بنية htp_gather_لیکې_params * محدود kparams,
    ثابت HVX_ویکټور * محدود tab, ثابت HVX_ویکټور * محدود inc, HVX_ویکټور * محدود cur) {
    ثابت uint32_t vg = kparams->vg;
    ثابت uint32_t region_اندازه = kparams->region_اندازه;
    لكل (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #لارښوونه unroll(4)
    لكل (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ویکټور *) (بهر + v * 128), (size_t) region, region_اندازه, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        إذا (++k == vg) {
            k = 0;
        }
    }
}

ساكن فراغ gather_لیکې_thread_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية gather_لیکې_task * task = (ثابت بنية gather_لیکې_task *) معلومات;
    ثابت بنية htp_gather_لیکې_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_دننه_واټن = MIN(ith * task->لیکې_فی_thread, task->لیکې);
    ثابت uint32_t r_beg    = task->r_پیل + r_دننه_واټن;
    ثابت uint32_t r_پای_ټکی    = task->r_پیل + MIN(r_دننه_واټن + task->لیکې_فی_thread, task->لیکې);
    إذا (r_beg >= r_پای_ټکی) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * بنسټ   = task->octx->ctx->vtcm_بنسټ + ith * kparams->ټوټه_2;
    HVX_ویکټور * tab = (HVX_ویکټور *) بنسټ;
    HVX_ویکټور * inc = tab + kparams->vg;
    HVX_ویکټور * cur = inc + kparams->vg;
    uint8_t * region_بنسټ = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_بنسټ, region_بنسټ + kparams->buf_stride };
    uint8_t * بهر[2]    = { region[0] + kparams->بهر_off, region[1] + kparams->بهر_off };

    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_لیکې_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_کتار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_دننه   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0    = MIN(nr_max, r_پای_ټکی - r_beg);

    dma_کتار_ټېل(q, dma_make_معلومات(task->dst, بهر[0]), 0, 0, 0, 0); // dummy out
    gather_لیکې_dma_دننه(q, task, region[0], r_beg, nr0, 4);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_پای_ټکی) {
        ثابت uint32_t nr1 = MIN(nr_max, r_پای_ټکی - r1);
        dma_کتار_ټېل(q, dma_make_معلومات(task->dst, بهر[1]), 0, 0, 0, 0); // dummy out
        gather_لیکې_dma_دننه(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_پای_ټکی; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_پای_ټکی - r);
        ثابت uint32_t بل_buf = buf ^ 1;

        dma_کتار_راوباسه(q); // complete out
        لكل (uint32_t i = 0; i < n_دننه; i++) {
            dma_کتار_راوباسه(q);
        }

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(بهر[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(بهر[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_لیکې_dma_بهر(q, task, بهر[buf], r, nr, 4);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_پای_ټکی) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_پای_ټکی - r_prefetch);
            gather_لیکې_dma_دننه(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = بل_buf;
    }
    dma_کتار_flush(q);
}

ساكن فراغ gather_لیکې_thread_f16(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية gather_لیکې_task * task = (ثابت بنية gather_لیکې_task *) معلومات;
    ثابت بنية htp_gather_لیکې_params * kparams = task->kparams;
    ثابت بنية htp_tensor * a = task->a;
    ثابت بنية htp_tensor * b = task->b;

    ثابت uint32_t r_دننه_واټن = MIN(ith * task->لیکې_فی_thread, task->لیکې);
    ثابت uint32_t r_beg    = task->r_پیل + r_دننه_واټن;
    ثابت uint32_t r_پای_ټکی    = task->r_پیل + MIN(r_دننه_واټن + task->لیکې_فی_thread, task->لیکې);
    إذا (r_beg >= r_پای_ټکی) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * بنسټ   = task->octx->ctx->vtcm_بنسټ + ith * kparams->ټوټه_2;
    HVX_ویکټور * tab = (HVX_ویکټور *) بنسټ;
    HVX_ویکټور * inc = tab + kparams->vg;
    HVX_ویکټور * cur = inc + kparams->vg;
    uint8_t * region_بنسټ = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_بنسټ, region_بنسټ + kparams->buf_stride };
    uint8_t * بهر[2]    = { region[0] + kparams->بهر_off, region[1] + kparams->بهر_off };

    htp_trace_event_پیل(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_لیکې_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_کتار * q = task->octx->ctx->dma[ith];
    ثابت uint32_t n_دننه = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ثابت uint32_t nr_max = kparams->nr_max;
    ثابت uint32_t nr0 = MIN(nr_max, r_پای_ټکی - r_beg);
    dma_کتار_ټېل(q, dma_make_معلومات(task->dst, بهر[0]), 0, 0, 0, 0); // dummy out
    gather_لیکې_dma_دننه(q, task, region[0], r_beg, nr0, 2);

    ثابت uint32_t r1 = r_beg + nr_max;
    إذا (r1 < r_پای_ټکی) {
        ثابت uint32_t nr1 = MIN(nr_max, r_پای_ټکی - r1);
        dma_کتار_ټېل(q, dma_make_معلومات(task->dst, بهر[1]), 0, 0, 0, 0); // dummy out
        gather_لیکې_dma_دننه(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    لكل (uint32_t r = r_beg; r < r_پای_ټکی; r += nr_max) {
        ثابت uint32_t nr = MIN(nr_max, r_پای_ټکی - r);
        ثابت uint32_t بل_buf = buf ^ 1;

        dma_کتار_راوباسه(q); // complete out
        لكل (uint32_t i = 0; i < n_دننه; i++) {
            dma_کتار_راوباسه(q);
        }

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ثابت uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(بهر[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(بهر[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_لیکې_dma_بهر(q, task, بهر[buf], r, nr, 2);

        ثابت uint32_t r_prefetch = r + 2 * nr_max;
        إذا (r_prefetch < r_پای_ټکی) {
            ثابت uint32_t nr_prefetch = MIN(nr_max, r_پای_ټکی - r_prefetch);
            gather_لیکې_dma_دننه(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = بل_buf;
    }
    dma_کتار_flush(q);
}

صحيح htp_gather_لیکې(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_لیکې_params * kparams,
    uint32_t n_تارونه) {

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    أكد(kparams->ټوټه_2 * octx->n_تارونه <= octx->ctx->vtcm_اندازه);

    ثابت uint32_t ټول_لیکې = kparams->ټول_لیکې;
    uint32_t r_پیل = 0;
    uint32_t r_شمېر = ټول_لیکې;
    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(
            ټول_لیکې, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        r_پیل = range.پیل;
        r_شمېر = range.شمېر;
    }

    إذا (r_شمېر == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t elem_اندازه = (a->type == HTP_TYPE_F16) ? 2 : 4;

    بنية gather_لیکې_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->معلومات;
    task.r_پیل         = r_پیل;
    task.لیکې            = r_شمېر;
    task.kparams         = kparams;
    task.لیکې_فی_thread = kparams->لیکې_فی_thread;

    إذا (elem_اندازه == 2) {
        task.compute = (kparams->vg == 1) ? gather_لیکې_compute_f16_vg1 : gather_لیکې_compute_f16_vgn;
    } وإلا {
        task.compute = (kparams->vg == 1) ? gather_لیکې_compute_f32_vg1 : gather_لیکې_compute_f32_vgn;
    }

    work_کتار_func_t کارګر_func = (elem_اندازه == 2) ? gather_لیکې_thread_f16 : gather_لیکې_thread_f32;
    work_کتار_پرلپسې(octx->ctx->work_کتار, کارګر_func, &task, octx->n_تارونه);
    أرجع HTP_حالت_OK;
}
