#समावेश "gather-rows.h"

#समावेश <assert.h>

#समावेश "hex-common.h"
#समावेश "hex-profile.h"
#समावेश "hvx-utils.h"
#समावेश "dma-queue.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "work-queue.h"

प्रकार व्योम (*gather_rows_compute_fn_t)(
    uint8_t * restrict out, स्थिर uint8_t * restrict region, uint32_t nvec,
    स्थिर काष्ठा htp_gather_rows_params * restrict kparams,
    स्थिर HVX_Vector * restrict tab, स्थिर HVX_Vector * restrict inc, HVX_Vector * restrict cur);

काष्ठा gather_rows_task {
    काष्ठा htp_ops_context *              octx;
    स्थिर काष्ठा htp_tensor *             a;
    स्थिर काष्ठा htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_start;
    uint32_t                              rows;
    uint32_t                              rows_per_thread;
    स्थिर काष्ठा htp_gather_rows_params * kparams;
    gather_rows_compute_fn_t              compute;
};

अटल अंतरभूत __attribute__((always_inline)) व्योम gather_rows_dma_in(
    dma_queue * q, स्थिर काष्ठा gather_rows_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_आकार) {
    स्थिर काष्ठा htp_tensor * a = task->a;
    स्थिर काष्ठा htp_tensor * b = task->b;
    स्थिर काष्ठा htp_gather_rows_params * kparams = task->kparams;

    स्थिर माप_प्रकार a_आकार = (nr - 1) * a->nb[1] + kparams->na * elem_आकार;
    dma_queue_धकेलो(q, dma_make_तथ्य(region, a->तथ्य + r * a->nb[1]), a_आकार, a_आकार, a_आकार, 1);
    अगर (b) {
        अगर (kparams->b_dense) {
            स्थिर माप_प्रकार b_आकार = (nr - 1) * b->nb[1] + kparams->nb * elem_आकार;
            dma_queue_धकेलो(q, dma_make_तथ्य(region + kparams->v1, b->तथ्य + r * b->nb[1]), b_आकार, b_आकार, b_आकार, 1);
        } अन्यथा {
            स्थिर माप_प्रकार b_आकार = (nr - 1) * b->nb[1] + elem_आकार;
            क्रम (uint32_t k = 0; k < kparams->nb; k++) {
                dma_queue_धकेलो(q, dma_make_तथ्य(region + kparams->v1 + k * kparams->span1,
                                                b->तथ्य + k * b->nb[0] + r * b->nb[1]), b_आकार, b_आकार, b_आकार, 1);
            }
        }
    }
}

अटल अंतरभूत व्योम gather_rows_dma_out(dma_queue * q, स्थिर काष्ठा gather_rows_task * task,
                                       uint8_t * out, uint32_t r, uint32_t nr, uint32_t elem_आकार) {
    स्थिर काष्ठा htp_gather_rows_params * kparams = task->kparams;
    स्थिर माप_प्रकार out_आकार = nr * kparams->ne * elem_आकार;
    dma_queue_धकेलो(q, dma_make_तथ्य(task->dst + r * kparams->ne * elem_आकार, out), out_आकार, out_आकार, out_आकार, 1);
}

अटल __attribute__((noinline)) व्योम gather_rows_init_tables_f32(
    HVX_Vector * tab, HVX_Vector * inc,
    स्थिर काष्ठा htp_gather_rows_params * kparams,
    स्थिर काष्ठा htp_tensor * a, स्थिर काष्ठा htp_tensor * b) {

    स्थिर int32_t s_a = kparams->rg * a->nb[1];
    स्थिर int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    स्थिर uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    क्रम (uint32_t k = 0; k < kparams->vg; k++) {
        क्रम (uint32_t lane = 0; lane < 32; lane++) {
            अगर (col < kparams->na) {
                t_buf[lane] = row * a->nb[1] + col * 4;
                s_buf[lane] = s_a;
            } अन्यथा {
                t_buf[lane] = kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1;
                s_buf[lane] = s_b;
            }
            अगर (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(स्थिर HVX_Vector *) t_buf;
        inc[k] = *(स्थिर HVX_Vector *) s_buf;
    }
}

अटल __attribute__((noinline)) व्योम gather_rows_init_tables_f16(
    HVX_Vector * tab, HVX_Vector * inc,
    स्थिर काष्ठा htp_gather_rows_params * kparams,
    स्थिर काष्ठा htp_tensor * a, स्थिर काष्ठा htp_tensor * b) {

    स्थिर int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    स्थिर int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    स्थिर uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    क्रम (uint32_t k = 0; k < kparams->vg; k++) {
        क्रम (uint32_t lane = 0; lane < 64; lane++) {
            अगर (col < kparams->na) {
                t_buf[lane] = (int16_t) (row * a->nb[1] + col * 2);
                s_buf[lane] = s_a;
            } अन्यथा {
                t_buf[lane] = (int16_t) (kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1);
                s_buf[lane] = s_b;
            }
            अगर (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(स्थिर HVX_Vector *) t_buf;
        inc[k] = *(स्थिर HVX_Vector *) s_buf;
    }
}

अटल व्योम gather_rows_compute_f32_vg1(
    uint8_t * restrict out, स्थिर uint8_t * restrict region, uint32_t nvec,
    स्थिर काष्ठा htp_gather_rows_params * restrict kparams,
    स्थिर HVX_Vector * restrict tab, स्थिर HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (व्योम) cur;
    स्थिर uint32_t region_आकार = kparams->region_आकार;
    HVX_Vector vcur = tab[0];
    स्थिर HVX_Vector vinc = inc[0];
    #आशय unroll(4)
    क्रम (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (माप_प्रकार) region, region_आकार, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

अटल व्योम gather_rows_compute_f32_vgn(
    uint8_t * restrict out, स्थिर uint8_t * restrict region, uint32_t nvec,
    स्थिर काष्ठा htp_gather_rows_params * restrict kparams,
    स्थिर HVX_Vector * restrict tab, स्थिर HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    स्थिर uint32_t vg = kparams->vg;
    स्थिर uint32_t region_आकार = kparams->region_आकार;
    क्रम (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #आशय unroll(4)
    क्रम (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (माप_प्रकार) region, region_आकार, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        अगर (++k == vg) {
            k = 0;
        }
    }
}

अटल व्योम gather_rows_compute_f16_vg1(
    uint8_t * restrict out, स्थिर uint8_t * restrict region, uint32_t nvec,
    स्थिर काष्ठा htp_gather_rows_params * restrict kparams,
    स्थिर HVX_Vector * restrict tab, स्थिर HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (व्योम) cur;
    स्थिर uint32_t region_आकार = kparams->region_आकार;
    HVX_Vector vcur = tab[0];
    स्थिर HVX_Vector vinc = inc[0];
    #आशय unroll(4)
    क्रम (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (माप_प्रकार) region, region_आकार, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

अटल व्योम gather_rows_compute_f16_vgn(
    uint8_t * restrict out, स्थिर uint8_t * restrict region, uint32_t nvec,
    स्थिर काष्ठा htp_gather_rows_params * restrict kparams,
    स्थिर HVX_Vector * restrict tab, स्थिर HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    स्थिर uint32_t vg = kparams->vg;
    स्थिर uint32_t region_आकार = kparams->region_आकार;
    क्रम (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #आशय unroll(4)
    क्रम (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (माप_प्रकार) region, region_आकार, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        अगर (++k == vg) {
            k = 0;
        }
    }
}

अटल व्योम gather_rows_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा gather_rows_task * task = (स्थिर काष्ठा gather_rows_task *) तथ्य;
    स्थिर काष्ठा htp_gather_rows_params * kparams = task->kparams;
    स्थिर काष्ठा htp_tensor * a = task->a;
    स्थिर काष्ठा htp_tensor * b = task->b;

    स्थिर uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    स्थिर uint32_t r_beg    = task->r_start + r_offset;
    स्थिर uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    अगर (r_beg >= r_end) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * base   = task->octx->ctx->vtcm_base + ith * kparams->slice;
    HVX_Vector * tab = (HVX_Vector *) base;
    HVX_Vector * inc = tab + kparams->vg;
    HVX_Vector * cur = inc + kparams->vg;
    uint8_t * region_base = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_base, region_base + kparams->buf_stride };
    uint8_t * out[2]    = { region[0] + kparams->out_off, region[1] + kparams->out_off };

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_rows_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];
    स्थिर uint32_t n_in   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    स्थिर uint32_t nr_max = kparams->nr_max;
    स्थिर uint32_t nr0    = MIN(nr_max, r_end - r_beg);

    dma_queue_धकेलो(q, dma_make_तथ्य(task->dst, out[0]), 0, 0, 0, 0); // dummy out
    gather_rows_dma_in(q, task, region[0], r_beg, nr0, 4);

    स्थिर uint32_t r1 = r_beg + nr_max;
    अगर (r1 < r_end) {
        स्थिर uint32_t nr1 = MIN(nr_max, r_end - r1);
        dma_queue_धकेलो(q, dma_make_तथ्य(task->dst, out[1]), 0, 0, 0, 0); // dummy out
        gather_rows_dma_in(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    क्रम (uint32_t r = r_beg; r < r_end; r += nr_max) {
        स्थिर uint32_t nr = MIN(nr_max, r_end - r);
        स्थिर uint32_t अग्र_buf = buf ^ 1;

        dma_queue_निकालो(q); // complete out
        क्रम (uint32_t i = 0; i < n_in; i++) {
            dma_queue_निकालो(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        स्थिर uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(out[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(out[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_rows_dma_out(q, task, out[buf], r, nr, 4);

        स्थिर uint32_t r_prefetch = r + 2 * nr_max;
        अगर (r_prefetch < r_end) {
            स्थिर uint32_t nr_prefetch = MIN(nr_max, r_end - r_prefetch);
            gather_rows_dma_in(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = अग्र_buf;
    }
    dma_queue_flush(q);
}

अटल व्योम gather_rows_thread_f16(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा gather_rows_task * task = (स्थिर काष्ठा gather_rows_task *) तथ्य;
    स्थिर काष्ठा htp_gather_rows_params * kparams = task->kparams;
    स्थिर काष्ठा htp_tensor * a = task->a;
    स्थिर काष्ठा htp_tensor * b = task->b;

    स्थिर uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    स्थिर uint32_t r_beg    = task->r_start + r_offset;
    स्थिर uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    अगर (r_beg >= r_end) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * base   = task->octx->ctx->vtcm_base + ith * kparams->slice;
    HVX_Vector * tab = (HVX_Vector *) base;
    HVX_Vector * inc = tab + kparams->vg;
    HVX_Vector * cur = inc + kparams->vg;
    uint8_t * region_base = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_base, region_base + kparams->buf_stride };
    uint8_t * out[2]    = { region[0] + kparams->out_off, region[1] + kparams->out_off };

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_rows_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];
    स्थिर uint32_t n_in = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    स्थिर uint32_t nr_max = kparams->nr_max;
    स्थिर uint32_t nr0 = MIN(nr_max, r_end - r_beg);
    dma_queue_धकेलो(q, dma_make_तथ्य(task->dst, out[0]), 0, 0, 0, 0); // dummy out
    gather_rows_dma_in(q, task, region[0], r_beg, nr0, 2);

    स्थिर uint32_t r1 = r_beg + nr_max;
    अगर (r1 < r_end) {
        स्थिर uint32_t nr1 = MIN(nr_max, r_end - r1);
        dma_queue_धकेलो(q, dma_make_तथ्य(task->dst, out[1]), 0, 0, 0, 0); // dummy out
        gather_rows_dma_in(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    क्रम (uint32_t r = r_beg; r < r_end; r += nr_max) {
        स्थिर uint32_t nr = MIN(nr_max, r_end - r);
        स्थिर uint32_t अग्र_buf = buf ^ 1;

        dma_queue_निकालो(q); // complete out
        क्रम (uint32_t i = 0; i < n_in; i++) {
            dma_queue_निकालो(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        स्थिर uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(out[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(out[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_rows_dma_out(q, task, out[buf], r, nr, 2);

        स्थिर uint32_t r_prefetch = r + 2 * nr_max;
        अगर (r_prefetch < r_end) {
            स्थिर uint32_t nr_prefetch = MIN(nr_max, r_end - r_prefetch);
            gather_rows_dma_in(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = अग्र_buf;
    }
    dma_queue_flush(q);
}

पूर्णांक htp_gather_rows(
    काष्ठा htp_ops_context * octx,
    स्थिर काष्ठा htp_tensor * a,
    स्थिर काष्ठा htp_tensor * b,
    स्थिर काष्ठा htp_gather_rows_params * kparams,
    uint32_t n_threads) {

    अगर (!htp_ops_context_set_n_threads(octx, n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    निश्चित(kparams->slice * octx->n_threads <= octx->ctx->vtcm_आकार);

    स्थिर uint32_t योग_rows = kparams->योग_rows;
    uint32_t r_start = 0;
    uint32_t r_गिनती = योग_rows;
    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            योग_rows, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        r_start = range.start;
        r_गिनती = range.गिनती;
    }

    अगर (r_गिनती == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t elem_आकार = (a->type == HTP_TYPE_F16) ? 2 : 4;

    काष्ठा gather_rows_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->तथ्य;
    task.r_start         = r_start;
    task.rows            = r_गिनती;
    task.kparams         = kparams;
    task.rows_per_thread = kparams->rows_per_thread;

    अगर (elem_आकार == 2) {
        task.compute = (kparams->vg == 1) ? gather_rows_compute_f16_vg1 : gather_rows_compute_f16_vgn;
    } अन्यथा {
        task.compute = (kparams->vg == 1) ? gather_rows_compute_f32_vg1 : gather_rows_compute_f32_vgn;
    }

    work_queue_func_t worker_func = (elem_आकार == 2) ? gather_rows_thread_f16 : gather_rows_thread_f32;
    work_queue_run(octx->ctx->work_queue, worker_func, &task, octx->n_threads);
    वापस HTP_STATUS_OK;
}
