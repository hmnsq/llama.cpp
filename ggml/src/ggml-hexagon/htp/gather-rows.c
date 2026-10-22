#ਅਨ੍ਤਰ੍ਭਾਵ "gather-rows.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*gather_ਪੰਕਤੀਆਂ_compute_fn_t)(
    uint8_t * ਸੀਮਿਤ ਬਾਹਰਲਾ, ਅਚਲ uint8_t * ਸੀਮਿਤ region, uint32_t nvec,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * ਸੀਮਿਤ kparams,
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tab, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ inc, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ cur);

ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task {
    ਬਣਤਰ htp_ops_context *              octx;
    ਅਚਲ ਬਣਤਰ htp_tensor *             a;
    ਅਚਲ ਬਣਤਰ htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_ਸ਼ੁਰੂ;
    uint32_t                              ਪੰਕਤੀਆਂ;
    uint32_t                              ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams;
    gather_ਪੰਕਤੀਆਂ_compute_fn_t              compute;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ __attribute__((always_inline)) ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_dma_ਵਿੱਚ(
    dma_ਕਤਾਰ * q, ਅਚਲ ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_ਮਾਪ) {
    ਅਚਲ ਬਣਤਰ htp_tensor * a = task->a;
    ਅਚਲ ਬਣਤਰ htp_tensor * b = task->b;
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams = task->kparams;

    ਅਚਲ size_t a_ਮਾਪ = (nr - 1) * a->nb[1] + kparams->na * elem_ਮਾਪ;
    dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(region, a->ਅੰਕੜੇ + r * a->nb[1]), a_ਮਾਪ, a_ਮਾਪ, a_ਮਾਪ, 1);
    ਜੇ (b) {
        ਜੇ (kparams->b_dense) {
            ਅਚਲ size_t b_ਮਾਪ = (nr - 1) * b->nb[1] + kparams->nb * elem_ਮਾਪ;
            dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(region + kparams->v1, b->ਅੰਕੜੇ + r * b->nb[1]), b_ਮਾਪ, b_ਮਾਪ, b_ਮਾਪ, 1);
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t b_ਮਾਪ = (nr - 1) * b->nb[1] + elem_ਮਾਪ;
            ਲਈ (uint32_t k = 0; k < kparams->nb; k++) {
                dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(region + kparams->v1 + k * kparams->span1,
                                                b->ਅੰਕੜੇ + k * b->nb[0] + r * b->nb[1]), b_ਮਾਪ, b_ਮਾਪ, b_ਮਾਪ, 1);
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_dma_ਬਾਹਰਲਾ(dma_ਕਤਾਰ * q, ਅਚਲ ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task * task,
                                       uint8_t * ਬਾਹਰਲਾ, uint32_t r, uint32_t nr, uint32_t elem_ਮਾਪ) {
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams = task->kparams;
    ਅਚਲ size_t ਬਾਹਰਲਾ_ਮਾਪ = nr * kparams->ne * elem_ਮਾਪ;
    dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(task->dst + r * kparams->ne * elem_ਮਾਪ, ਬਾਹਰਲਾ), ਬਾਹਰਲਾ_ਮਾਪ, ਬਾਹਰਲਾ_ਮਾਪ, ਬਾਹਰਲਾ_ਮਾਪ, 1);
}

ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_init_tables_f32(
    HVX_ਸਦਿਸ਼ * tab, HVX_ਸਦਿਸ਼ * inc,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams,
    ਅਚਲ ਬਣਤਰ htp_tensor * a, ਅਚਲ ਬਣਤਰ htp_tensor * b) {

    ਅਚਲ int32_t s_a = kparams->rg * a->nb[1];
    ਅਚਲ int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ਅਚਲ uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t ਪੰਕਤੀ = 0;
    uint32_t ਖਾਨਾ = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    ਲਈ (uint32_t k = 0; k < kparams->vg; k++) {
        ਲਈ (uint32_t lane = 0; lane < 32; lane++) {
            ਜੇ (ਖਾਨਾ < kparams->na) {
                t_buf[lane] = ਪੰਕਤੀ * a->nb[1] + ਖਾਨਾ * 4;
                s_buf[lane] = s_a;
            } ਨਹੀਂਤਾਂ {
                t_buf[lane] = kparams->v1 + (ਖਾਨਾ - kparams->na) * kparams->span1 + ਪੰਕਤੀ * b_nb1;
                s_buf[lane] = s_b;
            }
            ਜੇ (++ਖਾਨਾ == kparams->ne) {
                ਖਾਨਾ = 0;
                ਪੰਕਤੀ++;
            }
        }
        tab[k] = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) t_buf;
        inc[k] = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) s_buf;
    }
}

ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_init_tables_f16(
    HVX_ਸਦਿਸ਼ * tab, HVX_ਸਦਿਸ਼ * inc,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams,
    ਅਚਲ ਬਣਤਰ htp_tensor * a, ਅਚਲ ਬਣਤਰ htp_tensor * b) {

    ਅਚਲ int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ਅਚਲ int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ਅਚਲ uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t ਪੰਕਤੀ = 0;
    uint32_t ਖਾਨਾ = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    ਲਈ (uint32_t k = 0; k < kparams->vg; k++) {
        ਲਈ (uint32_t lane = 0; lane < 64; lane++) {
            ਜੇ (ਖਾਨਾ < kparams->na) {
                t_buf[lane] = (int16_t) (ਪੰਕਤੀ * a->nb[1] + ਖਾਨਾ * 2);
                s_buf[lane] = s_a;
            } ਨਹੀਂਤਾਂ {
                t_buf[lane] = (int16_t) (kparams->v1 + (ਖਾਨਾ - kparams->na) * kparams->span1 + ਪੰਕਤੀ * b_nb1);
                s_buf[lane] = s_b;
            }
            ਜੇ (++ਖਾਨਾ == kparams->ne) {
                ਖਾਨਾ = 0;
                ਪੰਕਤੀ++;
            }
        }
        tab[k] = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) t_buf;
        inc[k] = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) s_buf;
    }
}

ਸਥਿਰ ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_compute_f32_vg1(
    uint8_t * ਸੀਮਿਤ ਬਾਹਰਲਾ, ਅਚਲ uint8_t * ਸੀਮਿਤ region, uint32_t nvec,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * ਸੀਮਿਤ kparams,
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tab, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ inc, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ cur) {
    (ਖਾਲੀ) cur;
    ਅਚਲ uint32_t region_ਮਾਪ = kparams->region_ਮਾਪ;
    HVX_ਸਦਿਸ਼ vcur = tab[0];
    ਅਚਲ HVX_ਸਦਿਸ਼ vinc = inc[0];
    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ + v * 128), (size_t) region, region_ਮਾਪ, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

ਸਥਿਰ ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_compute_f32_vgn(
    uint8_t * ਸੀਮਿਤ ਬਾਹਰਲਾ, ਅਚਲ uint8_t * ਸੀਮਿਤ region, uint32_t nvec,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * ਸੀਮਿਤ kparams,
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tab, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ inc, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ cur) {
    ਅਚਲ uint32_t vg = kparams->vg;
    ਅਚਲ uint32_t region_ਮਾਪ = kparams->region_ਮਾਪ;
    ਲਈ (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ + v * 128), (size_t) region, region_ਮਾਪ, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        ਜੇ (++k == vg) {
            k = 0;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_compute_f16_vg1(
    uint8_t * ਸੀਮਿਤ ਬਾਹਰਲਾ, ਅਚਲ uint8_t * ਸੀਮਿਤ region, uint32_t nvec,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * ਸੀਮਿਤ kparams,
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tab, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ inc, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ cur) {
    (ਖਾਲੀ) cur;
    ਅਚਲ uint32_t region_ਮਾਪ = kparams->region_ਮਾਪ;
    HVX_ਸਦਿਸ਼ vcur = tab[0];
    ਅਚਲ HVX_ਸਦਿਸ਼ vinc = inc[0];
    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ + v * 128), (size_t) region, region_ਮਾਪ, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

ਸਥਿਰ ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_compute_f16_vgn(
    uint8_t * ਸੀਮਿਤ ਬਾਹਰਲਾ, ਅਚਲ uint8_t * ਸੀਮਿਤ region, uint32_t nvec,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * ਸੀਮਿਤ kparams,
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tab, ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ inc, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ cur) {
    ਅਚਲ uint32_t vg = kparams->vg;
    ਅਚਲ uint32_t region_ਮਾਪ = kparams->region_ਮਾਪ;
    ਲਈ (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ + v * 128), (size_t) region, region_ਮਾਪ, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        ਜੇ (++k == vg) {
            k = 0;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task * task = (ਅਚਲ ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task *) ਅੰਕੜੇ;
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams = task->kparams;
    ਅਚਲ ਬਣਤਰ htp_tensor * a = task->a;
    ਅਚਲ ਬਣਤਰ htp_tensor * b = task->b;

    ਅਚਲ uint32_t r_ਵਿੱਥ = MIN(ith * task->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, task->ਪੰਕਤੀਆਂ);
    ਅਚਲ uint32_t r_beg    = task->r_ਸ਼ੁਰੂ + r_ਵਿੱਥ;
    ਅਚਲ uint32_t r_ਸਿਰਾ    = task->r_ਸ਼ੁਰੂ + MIN(r_ਵਿੱਥ + task->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, task->ਪੰਕਤੀਆਂ);
    ਜੇ (r_beg >= r_ਸਿਰਾ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * ਅਧਾਰ   = task->octx->ctx->vtcm_ਅਧਾਰ + ith * kparams->ਟੁਕੜਾ_2;
    HVX_ਸਦਿਸ਼ * tab = (HVX_ਸਦਿਸ਼ *) ਅਧਾਰ;
    HVX_ਸਦਿਸ਼ * inc = tab + kparams->vg;
    HVX_ਸਦਿਸ਼ * cur = inc + kparams->vg;
    uint8_t * region_ਅਧਾਰ = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_ਅਧਾਰ, region_ਅਧਾਰ + kparams->buf_stride };
    uint8_t * ਬਾਹਰਲਾ[2]    = { region[0] + kparams->ਬਾਹਰਲਾ_off, region[1] + kparams->ਬਾਹਰਲਾ_off };

    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_ਪੰਕਤੀਆਂ_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_ਕਤਾਰ * q = task->octx->ctx->dma[ith];
    ਅਚਲ uint32_t n_ਵਿੱਚ   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ਅਚਲ uint32_t nr_max = kparams->nr_max;
    ਅਚਲ uint32_t nr0    = MIN(nr_max, r_ਸਿਰਾ - r_beg);

    dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(task->dst, ਬਾਹਰਲਾ[0]), 0, 0, 0, 0); // dummy out
    gather_ਪੰਕਤੀਆਂ_dma_ਵਿੱਚ(q, task, region[0], r_beg, nr0, 4);

    ਅਚਲ uint32_t r1 = r_beg + nr_max;
    ਜੇ (r1 < r_ਸਿਰਾ) {
        ਅਚਲ uint32_t nr1 = MIN(nr_max, r_ਸਿਰਾ - r1);
        dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(task->dst, ਬਾਹਰਲਾ[1]), 0, 0, 0, 0); // dummy out
        gather_ਪੰਕਤੀਆਂ_dma_ਵਿੱਚ(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    ਲਈ (uint32_t r = r_beg; r < r_ਸਿਰਾ; r += nr_max) {
        ਅਚਲ uint32_t nr = MIN(nr_max, r_ਸਿਰਾ - r);
        ਅਚਲ uint32_t ਅਗਲਾ_buf = buf ^ 1;

        dma_ਕਤਾਰ_ਕੱਢੋ(q); // complete out
        ਲਈ (uint32_t i = 0; i < n_ਵਿੱਚ; i++) {
            dma_ਕਤਾਰ_ਕੱਢੋ(q);
        }

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ਅਚਲ uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(ਬਾਹਰਲਾ[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(ਬਾਹਰਲਾ[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_ਪੰਕਤੀਆਂ_dma_ਬਾਹਰਲਾ(q, task, ਬਾਹਰਲਾ[buf], r, nr, 4);

        ਅਚਲ uint32_t r_prefetch = r + 2 * nr_max;
        ਜੇ (r_prefetch < r_ਸਿਰਾ) {
            ਅਚਲ uint32_t nr_prefetch = MIN(nr_max, r_ਸਿਰਾ - r_prefetch);
            gather_ਪੰਕਤੀਆਂ_dma_ਵਿੱਚ(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = ਅਗਲਾ_buf;
    }
    dma_ਕਤਾਰ_flush(q);
}

ਸਥਿਰ ਖਾਲੀ gather_ਪੰਕਤੀਆਂ_thread_f16(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task * task = (ਅਚਲ ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task *) ਅੰਕੜੇ;
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams = task->kparams;
    ਅਚਲ ਬਣਤਰ htp_tensor * a = task->a;
    ਅਚਲ ਬਣਤਰ htp_tensor * b = task->b;

    ਅਚਲ uint32_t r_ਵਿੱਥ = MIN(ith * task->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, task->ਪੰਕਤੀਆਂ);
    ਅਚਲ uint32_t r_beg    = task->r_ਸ਼ੁਰੂ + r_ਵਿੱਥ;
    ਅਚਲ uint32_t r_ਸਿਰਾ    = task->r_ਸ਼ੁਰੂ + MIN(r_ਵਿੱਥ + task->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, task->ਪੰਕਤੀਆਂ);
    ਜੇ (r_beg >= r_ਸਿਰਾ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * ਅਧਾਰ   = task->octx->ctx->vtcm_ਅਧਾਰ + ith * kparams->ਟੁਕੜਾ_2;
    HVX_ਸਦਿਸ਼ * tab = (HVX_ਸਦਿਸ਼ *) ਅਧਾਰ;
    HVX_ਸਦਿਸ਼ * inc = tab + kparams->vg;
    HVX_ਸਦਿਸ਼ * cur = inc + kparams->vg;
    uint8_t * region_ਅਧਾਰ = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_ਅਧਾਰ, region_ਅਧਾਰ + kparams->buf_stride };
    uint8_t * ਬਾਹਰਲਾ[2]    = { region[0] + kparams->ਬਾਹਰਲਾ_off, region[1] + kparams->ਬਾਹਰਲਾ_off };

    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_ਪੰਕਤੀਆਂ_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_ਕਤਾਰ * q = task->octx->ctx->dma[ith];
    ਅਚਲ uint32_t n_ਵਿੱਚ = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ਅਚਲ uint32_t nr_max = kparams->nr_max;
    ਅਚਲ uint32_t nr0 = MIN(nr_max, r_ਸਿਰਾ - r_beg);
    dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(task->dst, ਬਾਹਰਲਾ[0]), 0, 0, 0, 0); // dummy out
    gather_ਪੰਕਤੀਆਂ_dma_ਵਿੱਚ(q, task, region[0], r_beg, nr0, 2);

    ਅਚਲ uint32_t r1 = r_beg + nr_max;
    ਜੇ (r1 < r_ਸਿਰਾ) {
        ਅਚਲ uint32_t nr1 = MIN(nr_max, r_ਸਿਰਾ - r1);
        dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(task->dst, ਬਾਹਰਲਾ[1]), 0, 0, 0, 0); // dummy out
        gather_ਪੰਕਤੀਆਂ_dma_ਵਿੱਚ(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    ਲਈ (uint32_t r = r_beg; r < r_ਸਿਰਾ; r += nr_max) {
        ਅਚਲ uint32_t nr = MIN(nr_max, r_ਸਿਰਾ - r);
        ਅਚਲ uint32_t ਅਗਲਾ_buf = buf ^ 1;

        dma_ਕਤਾਰ_ਕੱਢੋ(q); // complete out
        ਲਈ (uint32_t i = 0; i < n_ਵਿੱਚ; i++) {
            dma_ਕਤਾਰ_ਕੱਢੋ(q);
        }

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ਅਚਲ uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(ਬਾਹਰਲਾ[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(ਬਾਹਰਲਾ[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_ਪੰਕਤੀਆਂ_dma_ਬਾਹਰਲਾ(q, task, ਬਾਹਰਲਾ[buf], r, nr, 2);

        ਅਚਲ uint32_t r_prefetch = r + 2 * nr_max;
        ਜੇ (r_prefetch < r_ਸਿਰਾ) {
            ਅਚਲ uint32_t nr_prefetch = MIN(nr_max, r_ਸਿਰਾ - r_prefetch);
            gather_ਪੰਕਤੀਆਂ_dma_ਵਿੱਚ(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = ਅਗਲਾ_buf;
    }
    dma_ਕਤਾਰ_flush(q);
}

ਪੂਰਨ htp_gather_ਪੰਕਤੀਆਂ(
    ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ htp_tensor * a,
    ਅਚਲ ਬਣਤਰ htp_tensor * b,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams,
    uint32_t n_ਧਾਗੇ) {

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(kparams->ਟੁਕੜਾ_2 * octx->n_ਧਾਗੇ <= octx->ctx->vtcm_ਮਾਪ);

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t r_ਸ਼ੁਰੂ = 0;
    uint32_t r_ਗਿਣਤੀ = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            ਕੁੱਲ_ਪੰਕਤੀਆਂ, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        r_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        r_ਗਿਣਤੀ = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (r_ਗਿਣਤੀ == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t elem_ਮਾਪ = (a->type == HTP_TYPE_F16) ? 2 : 4;

    ਬਣਤਰ gather_ਪੰਕਤੀਆਂ_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->ਅੰਕੜੇ;
    task.r_ਸ਼ੁਰੂ         = r_ਸ਼ੁਰੂ;
    task.ਪੰਕਤੀਆਂ            = r_ਗਿਣਤੀ;
    task.kparams         = kparams;
    task.ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = kparams->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;

    ਜੇ (elem_ਮਾਪ == 2) {
        task.compute = (kparams->vg == 1) ? gather_ਪੰਕਤੀਆਂ_compute_f16_vg1 : gather_ਪੰਕਤੀਆਂ_compute_f16_vgn;
    } ਨਹੀਂਤਾਂ {
        task.compute = (kparams->vg == 1) ? gather_ਪੰਕਤੀਆਂ_compute_f32_vg1 : gather_ਪੰਕਤੀਆਂ_compute_f32_vgn;
    }

    work_ਕਤਾਰ_func_t ਕਾਮਾ_func = (elem_ਮਾਪ == 2) ? gather_ਪੰਕਤੀਆਂ_thread_f16 : gather_ਪੰਕਤੀਆਂ_thread_f32;
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ਕਾਮਾ_func, &task, octx->n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
