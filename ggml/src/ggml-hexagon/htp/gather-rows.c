#অন্তর্ভাব "gather-rows.h"

#অন্তর্ভাব <assert.h>

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "work-queue.h"

প্রকারলেখ শূন্য (*gather_rows_compute_fn_t)(
    uint8_t * restrict out, ধ্রুবক uint8_t * restrict region, uint32_t nvec,
    ধ্রুবক কাঠামো htp_gather_rows_params * restrict kparams,
    ধ্রুবক HVX_Vector * restrict tab, ধ্রুবক HVX_Vector * restrict inc, HVX_Vector * restrict cur);

কাঠামো gather_rows_task {
    কাঠামো htp_ops_context *              octx;
    ধ্রুবক কাঠামো htp_tensor *             a;
    ধ্রুবক কাঠামো htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_start;
    uint32_t                              rows;
    uint32_t                              rows_per_thread;
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams;
    gather_rows_compute_fn_t              compute;
};

স্থির অন্তঃরেখ __attribute__((always_inline)) শূন্য gather_rows_dma_in(
    dma_queue * q, ধ্রুবক কাঠামো gather_rows_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_আকার) {
    ধ্রুবক কাঠামো htp_tensor * a = task->a;
    ধ্রুবক কাঠামো htp_tensor * b = task->b;
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams = task->kparams;

    ধ্রুবক size_t a_আকার = (nr - 1) * a->nb[1] + kparams->na * elem_আকার;
    dma_queue_ঠেলো(q, dma_make_তথ্য(region, a->তথ্য + r * a->nb[1]), a_আকার, a_আকার, a_আকার, 1);
    যদি (b) {
        যদি (kparams->b_dense) {
            ধ্রুবক size_t b_আকার = (nr - 1) * b->nb[1] + kparams->nb * elem_আকার;
            dma_queue_ঠেলো(q, dma_make_তথ্য(region + kparams->v1, b->তথ্য + r * b->nb[1]), b_আকার, b_আকার, b_আকার, 1);
        } অন্যথা {
            ধ্রুবক size_t b_আকার = (nr - 1) * b->nb[1] + elem_আকার;
            জন্য (uint32_t k = 0; k < kparams->nb; k++) {
                dma_queue_ঠেলো(q, dma_make_তথ্য(region + kparams->v1 + k * kparams->span1,
                                                b->তথ্য + k * b->nb[0] + r * b->nb[1]), b_আকার, b_আকার, b_আকার, 1);
            }
        }
    }
}

স্থির অন্তঃরেখ শূন্য gather_rows_dma_out(dma_queue * q, ধ্রুবক কাঠামো gather_rows_task * task,
                                       uint8_t * out, uint32_t r, uint32_t nr, uint32_t elem_আকার) {
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams = task->kparams;
    ধ্রুবক size_t out_আকার = nr * kparams->ne * elem_আকার;
    dma_queue_ঠেলো(q, dma_make_তথ্য(task->dst + r * kparams->ne * elem_আকার, out), out_আকার, out_আকার, out_আকার, 1);
}

স্থির __attribute__((noinline)) শূন্য gather_rows_init_tables_f32(
    HVX_Vector * tab, HVX_Vector * inc,
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams,
    ধ্রুবক কাঠামো htp_tensor * a, ধ্রুবক কাঠামো htp_tensor * b) {

    ধ্রুবক int32_t s_a = kparams->rg * a->nb[1];
    ধ্রুবক int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ধ্রুবক uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    জন্য (uint32_t k = 0; k < kparams->vg; k++) {
        জন্য (uint32_t lane = 0; lane < 32; lane++) {
            যদি (col < kparams->na) {
                t_buf[lane] = row * a->nb[1] + col * 4;
                s_buf[lane] = s_a;
            } অন্যথা {
                t_buf[lane] = kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1;
                s_buf[lane] = s_b;
            }
            যদি (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(ধ্রুবক HVX_Vector *) t_buf;
        inc[k] = *(ধ্রুবক HVX_Vector *) s_buf;
    }
}

স্থির __attribute__((noinline)) শূন্য gather_rows_init_tables_f16(
    HVX_Vector * tab, HVX_Vector * inc,
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams,
    ধ্রুবক কাঠামো htp_tensor * a, ধ্রুবক কাঠামো htp_tensor * b) {

    ধ্রুবক int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ধ্রুবক int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ধ্রুবক uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    জন্য (uint32_t k = 0; k < kparams->vg; k++) {
        জন্য (uint32_t lane = 0; lane < 64; lane++) {
            যদি (col < kparams->na) {
                t_buf[lane] = (int16_t) (row * a->nb[1] + col * 2);
                s_buf[lane] = s_a;
            } অন্যথা {
                t_buf[lane] = (int16_t) (kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1);
                s_buf[lane] = s_b;
            }
            যদি (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(ধ্রুবক HVX_Vector *) t_buf;
        inc[k] = *(ধ্রুবক HVX_Vector *) s_buf;
    }
}

স্থির শূন্য gather_rows_compute_f32_vg1(
    uint8_t * restrict out, ধ্রুবক uint8_t * restrict region, uint32_t nvec,
    ধ্রুবক কাঠামো htp_gather_rows_params * restrict kparams,
    ধ্রুবক HVX_Vector * restrict tab, ধ্রুবক HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (শূন্য) cur;
    ধ্রুবক uint32_t region_আকার = kparams->region_আকার;
    HVX_Vector vcur = tab[0];
    ধ্রুবক HVX_Vector vinc = inc[0];
    #pragma unroll(4)
    জন্য (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, region_আকার, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

স্থির শূন্য gather_rows_compute_f32_vgn(
    uint8_t * restrict out, ধ্রুবক uint8_t * restrict region, uint32_t nvec,
    ধ্রুবক কাঠামো htp_gather_rows_params * restrict kparams,
    ধ্রুবক HVX_Vector * restrict tab, ধ্রুবক HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    ধ্রুবক uint32_t vg = kparams->vg;
    ধ্রুবক uint32_t region_আকার = kparams->region_আকার;
    জন্য (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #pragma unroll(4)
    জন্য (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, region_আকার, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        যদি (++k == vg) {
            k = 0;
        }
    }
}

স্থির শূন্য gather_rows_compute_f16_vg1(
    uint8_t * restrict out, ধ্রুবক uint8_t * restrict region, uint32_t nvec,
    ধ্রুবক কাঠামো htp_gather_rows_params * restrict kparams,
    ধ্রুবক HVX_Vector * restrict tab, ধ্রুবক HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (শূন্য) cur;
    ধ্রুবক uint32_t region_আকার = kparams->region_আকার;
    HVX_Vector vcur = tab[0];
    ধ্রুবক HVX_Vector vinc = inc[0];
    #pragma unroll(4)
    জন্য (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, region_আকার, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

স্থির শূন্য gather_rows_compute_f16_vgn(
    uint8_t * restrict out, ধ্রুবক uint8_t * restrict region, uint32_t nvec,
    ধ্রুবক কাঠামো htp_gather_rows_params * restrict kparams,
    ধ্রুবক HVX_Vector * restrict tab, ধ্রুবক HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    ধ্রুবক uint32_t vg = kparams->vg;
    ধ্রুবক uint32_t region_আকার = kparams->region_আকার;
    জন্য (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #pragma unroll(4)
    জন্য (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, region_আকার, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        যদি (++k == vg) {
            k = 0;
        }
    }
}

স্থির শূন্য gather_rows_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    (শূন্য) nth;
    ধ্রুবক কাঠামো gather_rows_task * task = (ধ্রুবক কাঠামো gather_rows_task *) তথ্য;
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams = task->kparams;
    ধ্রুবক কাঠামো htp_tensor * a = task->a;
    ধ্রুবক কাঠামো htp_tensor * b = task->b;

    ধ্রুবক uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    ধ্রুবক uint32_t r_beg    = task->r_start + r_offset;
    ধ্রুবক uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    যদি (r_beg >= r_end) {
        ফেরত;
    }

    কাঠামো htp_thread_trace * tr = &task->octx->ctx->trace[ith];

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
    ধ্রুবক uint32_t n_in   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ধ্রুবক uint32_t nr_max = kparams->nr_max;
    ধ্রুবক uint32_t nr0    = MIN(nr_max, r_end - r_beg);

    dma_queue_ঠেলো(q, dma_make_তথ্য(task->dst, out[0]), 0, 0, 0, 0); // dummy out
    gather_rows_dma_in(q, task, region[0], r_beg, nr0, 4);

    ধ্রুবক uint32_t r1 = r_beg + nr_max;
    যদি (r1 < r_end) {
        ধ্রুবক uint32_t nr1 = MIN(nr_max, r_end - r1);
        dma_queue_ঠেলো(q, dma_make_তথ্য(task->dst, out[1]), 0, 0, 0, 0); // dummy out
        gather_rows_dma_in(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    জন্য (uint32_t r = r_beg; r < r_end; r += nr_max) {
        ধ্রুবক uint32_t nr = MIN(nr_max, r_end - r);
        ধ্রুবক uint32_t পরবর্তী_buf = buf ^ 1;

        dma_queue_বের_করো(q); // complete out
        জন্য (uint32_t i = 0; i < n_in; i++) {
            dma_queue_বের_করো(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ধ্রুবক uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(out[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(out[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_rows_dma_out(q, task, out[buf], r, nr, 4);

        ধ্রুবক uint32_t r_prefetch = r + 2 * nr_max;
        যদি (r_prefetch < r_end) {
            ধ্রুবক uint32_t nr_prefetch = MIN(nr_max, r_end - r_prefetch);
            gather_rows_dma_in(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = পরবর্তী_buf;
    }
    dma_queue_flush(q);
}

স্থির শূন্য gather_rows_thread_f16(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    (শূন্য) nth;
    ধ্রুবক কাঠামো gather_rows_task * task = (ধ্রুবক কাঠামো gather_rows_task *) তথ্য;
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams = task->kparams;
    ধ্রুবক কাঠামো htp_tensor * a = task->a;
    ধ্রুবক কাঠামো htp_tensor * b = task->b;

    ধ্রুবক uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    ধ্রুবক uint32_t r_beg    = task->r_start + r_offset;
    ধ্রুবক uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    যদি (r_beg >= r_end) {
        ফেরত;
    }

    কাঠামো htp_thread_trace * tr = &task->octx->ctx->trace[ith];

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
    ধ্রুবক uint32_t n_in = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ধ্রুবক uint32_t nr_max = kparams->nr_max;
    ধ্রুবক uint32_t nr0 = MIN(nr_max, r_end - r_beg);
    dma_queue_ঠেলো(q, dma_make_তথ্য(task->dst, out[0]), 0, 0, 0, 0); // dummy out
    gather_rows_dma_in(q, task, region[0], r_beg, nr0, 2);

    ধ্রুবক uint32_t r1 = r_beg + nr_max;
    যদি (r1 < r_end) {
        ধ্রুবক uint32_t nr1 = MIN(nr_max, r_end - r1);
        dma_queue_ঠেলো(q, dma_make_তথ্য(task->dst, out[1]), 0, 0, 0, 0); // dummy out
        gather_rows_dma_in(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    জন্য (uint32_t r = r_beg; r < r_end; r += nr_max) {
        ধ্রুবক uint32_t nr = MIN(nr_max, r_end - r);
        ধ্রুবক uint32_t পরবর্তী_buf = buf ^ 1;

        dma_queue_বের_করো(q); // complete out
        জন্য (uint32_t i = 0; i < n_in; i++) {
            dma_queue_বের_করো(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ধ্রুবক uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(out[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(out[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_rows_dma_out(q, task, out[buf], r, nr, 2);

        ধ্রুবক uint32_t r_prefetch = r + 2 * nr_max;
        যদি (r_prefetch < r_end) {
            ধ্রুবক uint32_t nr_prefetch = MIN(nr_max, r_end - r_prefetch);
            gather_rows_dma_in(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = পরবর্তী_buf;
    }
    dma_queue_flush(q);
}

পূর্ণসংখ্যা htp_gather_rows(
    কাঠামো htp_ops_context * octx,
    ধ্রুবক কাঠামো htp_tensor * a,
    ধ্রুবক কাঠামো htp_tensor * b,
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams,
    uint32_t n_threads) {

    যদি (!htp_ops_context_set_n_threads(octx, n_threads)) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    প্রতিজ্ঞা(kparams->slice * octx->n_threads <= octx->ctx->vtcm_আকার);

    ধ্রুবক uint32_t যোগ_rows = kparams->যোগ_rows;
    uint32_t r_start = 0;
    uint32_t r_গুনতি = যোগ_rows;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            যোগ_rows, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        r_start = range.start;
        r_গুনতি = range.গুনতি;
    }

    যদি (r_গুনতি == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t elem_আকার = (a->type == HTP_TYPE_F16) ? 2 : 4;

    কাঠামো gather_rows_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->তথ্য;
    task.r_start         = r_start;
    task.rows            = r_গুনতি;
    task.kparams         = kparams;
    task.rows_per_thread = kparams->rows_per_thread;

    যদি (elem_আকার == 2) {
        task.compute = (kparams->vg == 1) ? gather_rows_compute_f16_vg1 : gather_rows_compute_f16_vgn;
    } অন্যথা {
        task.compute = (kparams->vg == 1) ? gather_rows_compute_f32_vg1 : gather_rows_compute_f32_vgn;
    }

    work_queue_func_t worker_func = (elem_আকার == 2) ? gather_rows_thread_f16 : gather_rows_thread_f32;
    work_queue_run(octx->ctx->work_queue, worker_func, &task, octx->n_threads);
    ফেরত HTP_STATUS_OK;
}
