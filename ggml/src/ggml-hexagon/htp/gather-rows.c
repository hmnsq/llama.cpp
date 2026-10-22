#अन्तर्भाव "gather-rows.h"

#अन्तर्भाव <assert.h>

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "work-queue.h"

प्रकारनिर्देश शून्य (*gather_पङ्क्तिकाः_compute_fn_t)(
    uint8_t * सीमित बहिर्गमनम्, ध्रुव uint8_t * सीमित region, uint32_t nvec,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * सीमित kparams,
    ध्रुव HVX_सदिशम् * सीमित tab, ध्रुव HVX_सदिशम् * सीमित inc, HVX_सदिशम् * सीमित cur);

संरचना gather_पङ्क्तिकाः_task {
    संरचना htp_ops_context *              octx;
    ध्रुव संरचना htp_tensor *             a;
    ध्रुव संरचना htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_start;
    uint32_t                              पङ्क्तिकाः;
    uint32_t                              पङ्क्तिकाः_प्रत्येकम्_thread;
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams;
    gather_पङ्क्तिकाः_compute_fn_t              compute;
};

स्थिर अन्तःस्थ __attribute__((always_inline)) शून्य gather_पङ्क्तिकाः_dma_अन्तर्(
    dma_queue * q, ध्रुव संरचना gather_पङ्क्तिकाः_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_आकार) {
    ध्रुव संरचना htp_tensor * a = task->a;
    ध्रुव संरचना htp_tensor * b = task->b;
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams = task->kparams;

    ध्रुव size_t a_आकार = (nr - 1) * a->nb[1] + kparams->na * elem_आकार;
    dma_queue_प्रक्षिप(q, dma_make_दत्त(region, a->दत्त + r * a->nb[1]), a_आकार, a_आकार, a_आकार, 1);
    यदि (b) {
        यदि (kparams->b_dense) {
            ध्रुव size_t b_आकार = (nr - 1) * b->nb[1] + kparams->nb * elem_आकार;
            dma_queue_प्रक्षिप(q, dma_make_दत्त(region + kparams->v1, b->दत्त + r * b->nb[1]), b_आकार, b_आकार, b_आकार, 1);
        } अन्यथा {
            ध्रुव size_t b_आकार = (nr - 1) * b->nb[1] + elem_आकार;
            प्रति (uint32_t k = 0; k < kparams->nb; k++) {
                dma_queue_प्रक्षिप(q, dma_make_दत्त(region + kparams->v1 + k * kparams->span1,
                                                b->दत्त + k * b->nb[0] + r * b->nb[1]), b_आकार, b_आकार, b_आकार, 1);
            }
        }
    }
}

स्थिर अन्तःस्थ शून्य gather_पङ्क्तिकाः_dma_बहिर्गमनम्(dma_queue * q, ध्रुव संरचना gather_पङ्क्तिकाः_task * task,
                                       uint8_t * बहिर्गमनम्, uint32_t r, uint32_t nr, uint32_t elem_आकार) {
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams = task->kparams;
    ध्रुव size_t बहिर्गमनम्_आकार = nr * kparams->ne * elem_आकार;
    dma_queue_प्रक्षिप(q, dma_make_दत्त(task->dst + r * kparams->ne * elem_आकार, बहिर्गमनम्), बहिर्गमनम्_आकार, बहिर्गमनम्_आकार, बहिर्गमनम्_आकार, 1);
}

स्थिर __attribute__((noinline)) शून्य gather_पङ्क्तिकाः_init_tables_f32(
    HVX_सदिशम् * tab, HVX_सदिशम् * inc,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams,
    ध्रुव संरचना htp_tensor * a, ध्रुव संरचना htp_tensor * b) {

    ध्रुव int32_t s_a = kparams->rg * a->nb[1];
    ध्रुव int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ध्रुव uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    प्रति (uint32_t k = 0; k < kparams->vg; k++) {
        प्रति (uint32_t lane = 0; lane < 32; lane++) {
            यदि (col < kparams->na) {
                t_buf[lane] = row * a->nb[1] + col * 4;
                s_buf[lane] = s_a;
            } अन्यथा {
                t_buf[lane] = kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1;
                s_buf[lane] = s_b;
            }
            यदि (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(ध्रुव HVX_सदिशम् *) t_buf;
        inc[k] = *(ध्रुव HVX_सदिशम् *) s_buf;
    }
}

स्थिर __attribute__((noinline)) शून्य gather_पङ्क्तिकाः_init_tables_f16(
    HVX_सदिशम् * tab, HVX_सदिशम् * inc,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams,
    ध्रुव संरचना htp_tensor * a, ध्रुव संरचना htp_tensor * b) {

    ध्रुव int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ध्रुव int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ध्रुव uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    प्रति (uint32_t k = 0; k < kparams->vg; k++) {
        प्रति (uint32_t lane = 0; lane < 64; lane++) {
            यदि (col < kparams->na) {
                t_buf[lane] = (int16_t) (row * a->nb[1] + col * 2);
                s_buf[lane] = s_a;
            } अन्यथा {
                t_buf[lane] = (int16_t) (kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1);
                s_buf[lane] = s_b;
            }
            यदि (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(ध्रुव HVX_सदिशम् *) t_buf;
        inc[k] = *(ध्रुव HVX_सदिशम् *) s_buf;
    }
}

स्थिर शून्य gather_पङ्क्तिकाः_compute_f32_vg1(
    uint8_t * सीमित बहिर्गमनम्, ध्रुव uint8_t * सीमित region, uint32_t nvec,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * सीमित kparams,
    ध्रुव HVX_सदिशम् * सीमित tab, ध्रुव HVX_सदिशम् * सीमित inc, HVX_सदिशम् * सीमित cur) {
    (शून्य) cur;
    ध्रुव uint32_t region_आकार = kparams->region_आकार;
    HVX_सदिशम् vcur = tab[0];
    ध्रुव HVX_सदिशम् vinc = inc[0];
    #निर्देशः unroll(4)
    प्रति (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_सदिशम् *) (बहिर्गमनम् + v * 128), (size_t) region, region_आकार, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

स्थिर शून्य gather_पङ्क्तिकाः_compute_f32_vgn(
    uint8_t * सीमित बहिर्गमनम्, ध्रुव uint8_t * सीमित region, uint32_t nvec,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * सीमित kparams,
    ध्रुव HVX_सदिशम् * सीमित tab, ध्रुव HVX_सदिशम् * सीमित inc, HVX_सदिशम् * सीमित cur) {
    ध्रुव uint32_t vg = kparams->vg;
    ध्रुव uint32_t region_आकार = kparams->region_आकार;
    प्रति (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #निर्देशः unroll(4)
    प्रति (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_सदिशम् *) (बहिर्गमनम् + v * 128), (size_t) region, region_आकार, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        यदि (++k == vg) {
            k = 0;
        }
    }
}

स्थिर शून्य gather_पङ्क्तिकाः_compute_f16_vg1(
    uint8_t * सीमित बहिर्गमनम्, ध्रुव uint8_t * सीमित region, uint32_t nvec,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * सीमित kparams,
    ध्रुव HVX_सदिशम् * सीमित tab, ध्रुव HVX_सदिशम् * सीमित inc, HVX_सदिशम् * सीमित cur) {
    (शून्य) cur;
    ध्रुव uint32_t region_आकार = kparams->region_आकार;
    HVX_सदिशम् vcur = tab[0];
    ध्रुव HVX_सदिशम् vinc = inc[0];
    #निर्देशः unroll(4)
    प्रति (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_सदिशम् *) (बहिर्गमनम् + v * 128), (size_t) region, region_आकार, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

स्थिर शून्य gather_पङ्क्तिकाः_compute_f16_vgn(
    uint8_t * सीमित बहिर्गमनम्, ध्रुव uint8_t * सीमित region, uint32_t nvec,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * सीमित kparams,
    ध्रुव HVX_सदिशम् * सीमित tab, ध्रुव HVX_सदिशम् * सीमित inc, HVX_सदिशम् * सीमित cur) {
    ध्रुव uint32_t vg = kparams->vg;
    ध्रुव uint32_t region_आकार = kparams->region_आकार;
    प्रति (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #निर्देशः unroll(4)
    प्रति (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_सदिशम् *) (बहिर्गमनम् + v * 128), (size_t) region, region_आकार, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        यदि (++k == vg) {
            k = 0;
        }
    }
}

स्थिर शून्य gather_पङ्क्तिकाः_thread_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    ध्रुव संरचना gather_पङ्क्तिकाः_task * task = (ध्रुव संरचना gather_पङ्क्तिकाः_task *) दत्त;
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams = task->kparams;
    ध्रुव संरचना htp_tensor * a = task->a;
    ध्रुव संरचना htp_tensor * b = task->b;

    ध्रुव uint32_t r_विस्थापनम् = MIN(ith * task->पङ्क्तिकाः_प्रत्येकम्_thread, task->पङ्क्तिकाः);
    ध्रुव uint32_t r_beg    = task->r_start + r_विस्थापनम्;
    ध्रुव uint32_t r_अन्तः    = task->r_start + MIN(r_विस्थापनम् + task->पङ्क्तिकाः_प्रत्येकम्_thread, task->पङ्क्तिकाः);
    यदि (r_beg >= r_अन्तः) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * आधारः   = task->octx->ctx->vtcm_आधारः + ith * kparams->खण्डिका_2;
    HVX_सदिशम् * tab = (HVX_सदिशम् *) आधारः;
    HVX_सदिशम् * inc = tab + kparams->vg;
    HVX_सदिशम् * cur = inc + kparams->vg;
    uint8_t * region_आधारः = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_आधारः, region_आधारः + kparams->buf_stride };
    uint8_t * बहिर्गमनम्[2]    = { region[0] + kparams->बहिर्गमनम्_off, region[1] + kparams->बहिर्गमनम्_off };

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_पङ्क्तिकाः_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];
    ध्रुव uint32_t n_अन्तर्   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ध्रुव uint32_t nr_max = kparams->nr_max;
    ध्रुव uint32_t nr0    = MIN(nr_max, r_अन्तः - r_beg);

    dma_queue_प्रक्षिप(q, dma_make_दत्त(task->dst, बहिर्गमनम्[0]), 0, 0, 0, 0); // dummy out
    gather_पङ्क्तिकाः_dma_अन्तर्(q, task, region[0], r_beg, nr0, 4);

    ध्रुव uint32_t r1 = r_beg + nr_max;
    यदि (r1 < r_अन्तः) {
        ध्रुव uint32_t nr1 = MIN(nr_max, r_अन्तः - r1);
        dma_queue_प्रक्षिप(q, dma_make_दत्त(task->dst, बहिर्गमनम्[1]), 0, 0, 0, 0); // dummy out
        gather_पङ्क्तिकाः_dma_अन्तर्(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    प्रति (uint32_t r = r_beg; r < r_अन्तः; r += nr_max) {
        ध्रुव uint32_t nr = MIN(nr_max, r_अन्तः - r);
        ध्रुव uint32_t अग्र_buf = buf ^ 1;

        dma_queue_निष्कासय(q); // complete out
        प्रति (uint32_t i = 0; i < n_अन्तर्; i++) {
            dma_queue_निष्कासय(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ध्रुव uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(बहिर्गमनम्[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(बहिर्गमनम्[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_पङ्क्तिकाः_dma_बहिर्गमनम्(q, task, बहिर्गमनम्[buf], r, nr, 4);

        ध्रुव uint32_t r_prefetch = r + 2 * nr_max;
        यदि (r_prefetch < r_अन्तः) {
            ध्रुव uint32_t nr_prefetch = MIN(nr_max, r_अन्तः - r_prefetch);
            gather_पङ्क्तिकाः_dma_अन्तर्(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = अग्र_buf;
    }
    dma_queue_flush(q);
}

स्थिर शून्य gather_पङ्क्तिकाः_thread_f16(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    ध्रुव संरचना gather_पङ्क्तिकाः_task * task = (ध्रुव संरचना gather_पङ्क्तिकाः_task *) दत्त;
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams = task->kparams;
    ध्रुव संरचना htp_tensor * a = task->a;
    ध्रुव संरचना htp_tensor * b = task->b;

    ध्रुव uint32_t r_विस्थापनम् = MIN(ith * task->पङ्क्तिकाः_प्रत्येकम्_thread, task->पङ्क्तिकाः);
    ध्रुव uint32_t r_beg    = task->r_start + r_विस्थापनम्;
    ध्रुव uint32_t r_अन्तः    = task->r_start + MIN(r_विस्थापनम् + task->पङ्क्तिकाः_प्रत्येकम्_thread, task->पङ्क्तिकाः);
    यदि (r_beg >= r_अन्तः) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * आधारः   = task->octx->ctx->vtcm_आधारः + ith * kparams->खण्डिका_2;
    HVX_सदिशम् * tab = (HVX_सदिशम् *) आधारः;
    HVX_सदिशम् * inc = tab + kparams->vg;
    HVX_सदिशम् * cur = inc + kparams->vg;
    uint8_t * region_आधारः = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_आधारः, region_आधारः + kparams->buf_stride };
    uint8_t * बहिर्गमनम्[2]    = { region[0] + kparams->बहिर्गमनम्_off, region[1] + kparams->बहिर्गमनम्_off };

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_पङ्क्तिकाः_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];
    ध्रुव uint32_t n_अन्तर् = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ध्रुव uint32_t nr_max = kparams->nr_max;
    ध्रुव uint32_t nr0 = MIN(nr_max, r_अन्तः - r_beg);
    dma_queue_प्रक्षिप(q, dma_make_दत्त(task->dst, बहिर्गमनम्[0]), 0, 0, 0, 0); // dummy out
    gather_पङ्क्तिकाः_dma_अन्तर्(q, task, region[0], r_beg, nr0, 2);

    ध्रुव uint32_t r1 = r_beg + nr_max;
    यदि (r1 < r_अन्तः) {
        ध्रुव uint32_t nr1 = MIN(nr_max, r_अन्तः - r1);
        dma_queue_प्रक्षिप(q, dma_make_दत्त(task->dst, बहिर्गमनम्[1]), 0, 0, 0, 0); // dummy out
        gather_पङ्क्तिकाः_dma_अन्तर्(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    प्रति (uint32_t r = r_beg; r < r_अन्तः; r += nr_max) {
        ध्रुव uint32_t nr = MIN(nr_max, r_अन्तः - r);
        ध्रुव uint32_t अग्र_buf = buf ^ 1;

        dma_queue_निष्कासय(q); // complete out
        प्रति (uint32_t i = 0; i < n_अन्तर्; i++) {
            dma_queue_निष्कासय(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ध्रुव uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(बहिर्गमनम्[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(बहिर्गमनम्[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_पङ्क्तिकाः_dma_बहिर्गमनम्(q, task, बहिर्गमनम्[buf], r, nr, 2);

        ध्रुव uint32_t r_prefetch = r + 2 * nr_max;
        यदि (r_prefetch < r_अन्तः) {
            ध्रुव uint32_t nr_prefetch = MIN(nr_max, r_अन्तः - r_prefetch);
            gather_पङ्क्तिकाः_dma_अन्तर्(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = अग्र_buf;
    }
    dma_queue_flush(q);
}

पूर्णाङ्क htp_gather_पङ्क्तिकाः(
    संरचना htp_ops_context * octx,
    ध्रुव संरचना htp_tensor * a,
    ध्रुव संरचना htp_tensor * b,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams,
    uint32_t n_तन्तवः) {

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    प्रतिज्ञा(kparams->खण्डिका_2 * octx->n_तन्तवः <= octx->ctx->vtcm_आकार);

    ध्रुव uint32_t योग_पङ्क्तिकाः = kparams->योग_पङ्क्तिकाः;
    uint32_t r_start = 0;
    uint32_t r_गणनम् = योग_पङ्क्तिकाः;
    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            योग_पङ्क्तिकाः, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        r_start = range.start;
        r_गणनम् = range.गणनम्;
    }

    यदि (r_गणनम् == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t elem_आकार = (a->type == HTP_TYPE_F16) ? 2 : 4;

    संरचना gather_पङ्क्तिकाः_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->दत्त;
    task.r_start         = r_start;
    task.पङ्क्तिकाः            = r_गणनम्;
    task.kparams         = kparams;
    task.पङ्क्तिकाः_प्रत्येकम्_thread = kparams->पङ्क्तिकाः_प्रत्येकम्_thread;

    यदि (elem_आकार == 2) {
        task.compute = (kparams->vg == 1) ? gather_पङ्क्तिकाः_compute_f16_vg1 : gather_पङ्क्तिकाः_compute_f16_vgn;
    } अन्यथा {
        task.compute = (kparams->vg == 1) ? gather_पङ्क्तिकाः_compute_f32_vg1 : gather_पङ्क्तिकाः_compute_f32_vgn;
    }

    work_queue_func_t कार्यकर्ता_func = (elem_आकार == 2) ? gather_पङ्क्तिकाः_thread_f16 : gather_पङ्क्तिकाः_thread_f32;
    work_queue_सततम्(octx->ctx->work_queue, कार्यकर्ता_func, &task, octx->n_तन्तवः);
    प्रत्यागम HTP_स्थितिः_OK;
}
