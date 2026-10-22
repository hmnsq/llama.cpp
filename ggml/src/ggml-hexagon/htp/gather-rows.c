#ಅನ್ತರ್ಭಾವ "gather-rows.h"

#ಅನ್ತರ್ಭಾವ <assert.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*gather_ಅಡ್ಡಸಾಲುಗಳು_compute_fn_t)(
    uint8_t * ಸೀಮಿತ ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ region, uint32_t nvec,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * ಸೀಮಿತ kparams,
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ tab, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ inc, HVX_ಸದಿಶ * ಸೀಮಿತ cur);

ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task {
    ರಚನೆ htp_ops_context *              octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *             a;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_ಆರಂಭ;
    uint32_t                              ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t                              ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams;
    gather_ಅಡ್ಡಸಾಲುಗಳು_compute_fn_t              compute;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ __attribute__((always_inline)) ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಒಳ(
    dma_ಸರತಿ * q, ಸ್ಥಿರಾಂಕ ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * a = task->a;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * b = task->b;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams = task->kparams;

    ಸ್ಥಿರಾಂಕ size_t a_ಗಾತ್ರ = (nr - 1) * a->nb[1] + kparams->na * elem_ಗಾತ್ರ;
    dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(region, a->ದತ್ತಾಂಶ + r * a->nb[1]), a_ಗಾತ್ರ, a_ಗಾತ್ರ, a_ಗಾತ್ರ, 1);
    ಆದರೆ (b) {
        ಆದರೆ (kparams->b_dense) {
            ಸ್ಥಿರಾಂಕ size_t b_ಗಾತ್ರ = (nr - 1) * b->nb[1] + kparams->nb * elem_ಗಾತ್ರ;
            dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(region + kparams->v1, b->ದತ್ತಾಂಶ + r * b->nb[1]), b_ಗಾತ್ರ, b_ಗಾತ್ರ, b_ಗಾತ್ರ, 1);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t b_ಗಾತ್ರ = (nr - 1) * b->nb[1] + elem_ಗಾತ್ರ;
            ಗಾಗಿ (uint32_t k = 0; k < kparams->nb; k++) {
                dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(region + kparams->v1 + k * kparams->span1,
                                                b->ದತ್ತಾಂಶ + k * b->nb[0] + r * b->nb[1]), b_ಗಾತ್ರ, b_ಗಾತ್ರ, b_ಗಾತ್ರ, 1);
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಹೊರಹೋಗು(dma_ಸರತಿ * q, ಸ್ಥಿರಾಂಕ ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task * task,
                                       uint8_t * ಹೊರಹೋಗು, uint32_t r, uint32_t nr, uint32_t elem_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams = task->kparams;
    ಸ್ಥಿರಾಂಕ size_t ಹೊರಹೋಗು_ಗಾತ್ರ = nr * kparams->ne * elem_ಗಾತ್ರ;
    dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(task->dst + r * kparams->ne * elem_ಗಾತ್ರ, ಹೊರಹೋಗು), ಹೊರಹೋಗು_ಗಾತ್ರ, ಹೊರಹೋಗು_ಗಾತ್ರ, ಹೊರಹೋಗು_ಗಾತ್ರ, 1);
}

ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_init_tables_f32(
    HVX_ಸದಿಶ * tab, HVX_ಸದಿಶ * inc,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * a, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * b) {

    ಸ್ಥಿರಾಂಕ int32_t s_a = kparams->rg * a->nb[1];
    ಸ್ಥಿರಾಂಕ int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    ಸ್ಥಿರಾಂಕ uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t ಅಡ್ಡಸಾಲು = 0;
    uint32_t ಲಂಬಸಾಲು = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    ಗಾಗಿ (uint32_t k = 0; k < kparams->vg; k++) {
        ಗಾಗಿ (uint32_t lane = 0; lane < 32; lane++) {
            ಆದರೆ (ಲಂಬಸಾಲು < kparams->na) {
                t_buf[lane] = ಅಡ್ಡಸಾಲು * a->nb[1] + ಲಂಬಸಾಲು * 4;
                s_buf[lane] = s_a;
            } ಇಲ್ಲದಿದ್ದರೆ {
                t_buf[lane] = kparams->v1 + (ಲಂಬಸಾಲು - kparams->na) * kparams->span1 + ಅಡ್ಡಸಾಲು * b_nb1;
                s_buf[lane] = s_b;
            }
            ಆದರೆ (++ಲಂಬಸಾಲು == kparams->ne) {
                ಲಂಬಸಾಲು = 0;
                ಅಡ್ಡಸಾಲು++;
            }
        }
        tab[k] = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) t_buf;
        inc[k] = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) s_buf;
    }
}

ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_init_tables_f16(
    HVX_ಸದಿಶ * tab, HVX_ಸದಿಶ * inc,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * a, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * b) {

    ಸ್ಥಿರಾಂಕ int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    ಸ್ಥಿರಾಂಕ int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    ಸ್ಥಿರಾಂಕ uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t ಅಡ್ಡಸಾಲು = 0;
    uint32_t ಲಂಬಸಾಲು = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    ಗಾಗಿ (uint32_t k = 0; k < kparams->vg; k++) {
        ಗಾಗಿ (uint32_t lane = 0; lane < 64; lane++) {
            ಆದರೆ (ಲಂಬಸಾಲು < kparams->na) {
                t_buf[lane] = (int16_t) (ಅಡ್ಡಸಾಲು * a->nb[1] + ಲಂಬಸಾಲು * 2);
                s_buf[lane] = s_a;
            } ಇಲ್ಲದಿದ್ದರೆ {
                t_buf[lane] = (int16_t) (kparams->v1 + (ಲಂಬಸಾಲು - kparams->na) * kparams->span1 + ಅಡ್ಡಸಾಲು * b_nb1);
                s_buf[lane] = s_b;
            }
            ಆದರೆ (++ಲಂಬಸಾಲು == kparams->ne) {
                ಲಂಬಸಾಲು = 0;
                ಅಡ್ಡಸಾಲು++;
            }
        }
        tab[k] = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) t_buf;
        inc[k] = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) s_buf;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f32_vg1(
    uint8_t * ಸೀಮಿತ ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ region, uint32_t nvec,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * ಸೀಮಿತ kparams,
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ tab, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ inc, HVX_ಸದಿಶ * ಸೀಮಿತ cur) {
    (ಶೂನ್ಯ) cur;
    ಸ್ಥಿರಾಂಕ uint32_t region_ಗಾತ್ರ = kparams->region_ಗಾತ್ರ;
    HVX_ಸದಿಶ vcur = tab[0];
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vinc = inc[0];
    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು + v * 128), (size_t) region, region_ಗಾತ್ರ, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f32_vgn(
    uint8_t * ಸೀಮಿತ ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ region, uint32_t nvec,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * ಸೀಮಿತ kparams,
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ tab, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ inc, HVX_ಸದಿಶ * ಸೀಮಿತ cur) {
    ಸ್ಥಿರಾಂಕ uint32_t vg = kparams->vg;
    ಸ್ಥಿರಾಂಕ uint32_t region_ಗಾತ್ರ = kparams->region_ಗಾತ್ರ;
    ಗಾಗಿ (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು + v * 128), (size_t) region, region_ಗಾತ್ರ, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        ಆದರೆ (++k == vg) {
            k = 0;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f16_vg1(
    uint8_t * ಸೀಮಿತ ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ region, uint32_t nvec,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * ಸೀಮಿತ kparams,
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ tab, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ inc, HVX_ಸದಿಶ * ಸೀಮಿತ cur) {
    (ಶೂನ್ಯ) cur;
    ಸ್ಥಿರಾಂಕ uint32_t region_ಗಾತ್ರ = kparams->region_ಗಾತ್ರ;
    HVX_ಸದಿಶ vcur = tab[0];
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vinc = inc[0];
    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು + v * 128), (size_t) region, region_ಗಾತ್ರ, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f16_vgn(
    uint8_t * ಸೀಮಿತ ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ region, uint32_t nvec,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * ಸೀಮಿತ kparams,
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ tab, ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ inc, HVX_ಸದಿಶ * ಸೀಮಿತ cur) {
    ಸ್ಥಿರಾಂಕ uint32_t vg = kparams->vg;
    ಸ್ಥಿರಾಂಕ uint32_t region_ಗಾತ್ರ = kparams->region_ಗಾತ್ರ;
    ಗಾಗಿ (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು + v * 128), (size_t) region, region_ಗಾತ್ರ, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        ಆದರೆ (++k == vg) {
            k = 0;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task * task = (ಸ್ಥಿರಾಂಕ ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams = task->kparams;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * a = task->a;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * b = task->b;

    ಸ್ಥಿರಾಂಕ uint32_t r_ಸ್ಥಾನಾಂತರ = MIN(ith * task->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, task->ಅಡ್ಡಸಾಲುಗಳು);
    ಸ್ಥಿರಾಂಕ uint32_t r_beg    = task->r_ಆರಂಭ + r_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint32_t r_ಕೊನೆ    = task->r_ಆರಂಭ + MIN(r_ಸ್ಥಾನಾಂತರ + task->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, task->ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (r_beg >= r_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * ತಳ   = task->octx->ctx->vtcm_ತಳ + ith * kparams->ತುಂಡು_2;
    HVX_ಸದಿಶ * tab = (HVX_ಸದಿಶ *) ತಳ;
    HVX_ಸದಿಶ * inc = tab + kparams->vg;
    HVX_ಸದಿಶ * cur = inc + kparams->vg;
    uint8_t * region_ತಳ = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_ತಳ, region_ತಳ + kparams->buf_stride };
    uint8_t * ಹೊರಹೋಗು[2]    = { region[0] + kparams->ಹೊರಹೋಗು_off, region[1] + kparams->ಹೊರಹೋಗು_off };

    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_ಅಡ್ಡಸಾಲುಗಳು_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_ಸರತಿ * q = task->octx->ctx->dma[ith];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಒಳ   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    ಸ್ಥಿರಾಂಕ uint32_t nr_max = kparams->nr_max;
    ಸ್ಥಿರಾಂಕ uint32_t nr0    = MIN(nr_max, r_ಕೊನೆ - r_beg);

    dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(task->dst, ಹೊರಹೋಗು[0]), 0, 0, 0, 0); // dummy out
    gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಒಳ(q, task, region[0], r_beg, nr0, 4);

    ಸ್ಥಿರಾಂಕ uint32_t r1 = r_beg + nr_max;
    ಆದರೆ (r1 < r_ಕೊನೆ) {
        ಸ್ಥಿರಾಂಕ uint32_t nr1 = MIN(nr_max, r_ಕೊನೆ - r1);
        dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(task->dst, ಹೊರಹೋಗು[1]), 0, 0, 0, 0); // dummy out
        gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಒಳ(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    ಗಾಗಿ (uint32_t r = r_beg; r < r_ಕೊನೆ; r += nr_max) {
        ಸ್ಥಿರಾಂಕ uint32_t nr = MIN(nr_max, r_ಕೊನೆ - r);
        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_buf = buf ^ 1;

        dma_ಸರತಿ_ತೆಗೆ(q); // complete out
        ಗಾಗಿ (uint32_t i = 0; i < n_ಒಳ; i++) {
            dma_ಸರತಿ_ತೆಗೆ(q);
        }

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ಸ್ಥಿರಾಂಕ uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(ಹೊರಹೋಗು[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(ಹೊರಹೋಗು[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಹೊರಹೋಗು(q, task, ಹೊರಹೋಗು[buf], r, nr, 4);

        ಸ್ಥಿರಾಂಕ uint32_t r_prefetch = r + 2 * nr_max;
        ಆದರೆ (r_prefetch < r_ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t nr_prefetch = MIN(nr_max, r_ಕೊನೆ - r_prefetch);
            gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಒಳ(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = ಮುಂದಿನ_buf;
    }
    dma_ಸರತಿ_flush(q);
}

ಸ್ಥಿರ ಶೂನ್ಯ gather_ಅಡ್ಡಸಾಲುಗಳು_thread_f16(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task * task = (ಸ್ಥಿರಾಂಕ ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams = task->kparams;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * a = task->a;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * b = task->b;

    ಸ್ಥಿರಾಂಕ uint32_t r_ಸ್ಥಾನಾಂತರ = MIN(ith * task->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, task->ಅಡ್ಡಸಾಲುಗಳು);
    ಸ್ಥಿರಾಂಕ uint32_t r_beg    = task->r_ಆರಂಭ + r_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint32_t r_ಕೊನೆ    = task->r_ಆರಂಭ + MIN(r_ಸ್ಥಾನಾಂತರ + task->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, task->ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (r_beg >= r_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * ತಳ   = task->octx->ctx->vtcm_ತಳ + ith * kparams->ತುಂಡು_2;
    HVX_ಸದಿಶ * tab = (HVX_ಸದಿಶ *) ತಳ;
    HVX_ಸದಿಶ * inc = tab + kparams->vg;
    HVX_ಸದಿಶ * cur = inc + kparams->vg;
    uint8_t * region_ತಳ = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_ತಳ, region_ತಳ + kparams->buf_stride };
    uint8_t * ಹೊರಹೋಗು[2]    = { region[0] + kparams->ಹೊರಹೋಗು_off, region[1] + kparams->ಹೊರಹೋಗು_off };

    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_ಅಡ್ಡಸಾಲುಗಳು_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_ಸರತಿ * q = task->octx->ctx->dma[ith];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಒಳ = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    ಸ್ಥಿರಾಂಕ uint32_t nr_max = kparams->nr_max;
    ಸ್ಥಿರಾಂಕ uint32_t nr0 = MIN(nr_max, r_ಕೊನೆ - r_beg);
    dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(task->dst, ಹೊರಹೋಗು[0]), 0, 0, 0, 0); // dummy out
    gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಒಳ(q, task, region[0], r_beg, nr0, 2);

    ಸ್ಥಿರಾಂಕ uint32_t r1 = r_beg + nr_max;
    ಆದರೆ (r1 < r_ಕೊನೆ) {
        ಸ್ಥಿರಾಂಕ uint32_t nr1 = MIN(nr_max, r_ಕೊನೆ - r1);
        dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(task->dst, ಹೊರಹೋಗು[1]), 0, 0, 0, 0); // dummy out
        gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಒಳ(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    ಗಾಗಿ (uint32_t r = r_beg; r < r_ಕೊನೆ; r += nr_max) {
        ಸ್ಥಿರಾಂಕ uint32_t nr = MIN(nr_max, r_ಕೊನೆ - r);
        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_buf = buf ^ 1;

        dma_ಸರತಿ_ತೆಗೆ(q); // complete out
        ಗಾಗಿ (uint32_t i = 0; i < n_ಒಳ; i++) {
            dma_ಸರತಿ_ತೆಗೆ(q);
        }

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        ಸ್ಥಿರಾಂಕ uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(ಹೊರಹೋಗು[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(ಹೊರಹೋಗು[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಹೊರಹೋಗು(q, task, ಹೊರಹೋಗು[buf], r, nr, 2);

        ಸ್ಥಿರಾಂಕ uint32_t r_prefetch = r + 2 * nr_max;
        ಆದರೆ (r_prefetch < r_ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t nr_prefetch = MIN(nr_max, r_ಕೊನೆ - r_prefetch);
            gather_ಅಡ್ಡಸಾಲುಗಳು_dma_ಒಳ(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = ಮುಂದಿನ_buf;
    }
    dma_ಸರತಿ_flush(q);
}

ಪೂರ್ಣಾಂಕ htp_gather_ಅಡ್ಡಸಾಲುಗಳು(
    ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * a,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * b,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams,
    uint32_t n_ಎಳೆಗಳು) {

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಪ್ರತಿಜ್ಞಾ(kparams->ತುಂಡು_2 * octx->n_ಎಳೆಗಳು <= octx->ctx->vtcm_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t r_ಆರಂಭ = 0;
    uint32_t r_ಎಣಿಕೆ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        r_ಆರಂಭ = range.ಆರಂಭ;
        r_ಎಣಿಕೆ = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (r_ಎಣಿಕೆ == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = (a->type == HTP_TYPE_F16) ? 2 : 4;

    ರಚನೆ gather_ಅಡ್ಡಸಾಲುಗಳು_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->ದತ್ತಾಂಶ;
    task.r_ಆರಂಭ         = r_ಆರಂಭ;
    task.ಅಡ್ಡಸಾಲುಗಳು            = r_ಎಣಿಕೆ;
    task.kparams         = kparams;
    task.ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = kparams->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;

    ಆದರೆ (elem_ಗಾತ್ರ == 2) {
        task.compute = (kparams->vg == 1) ? gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f16_vg1 : gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f16_vgn;
    } ಇಲ್ಲದಿದ್ದರೆ {
        task.compute = (kparams->vg == 1) ? gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f32_vg1 : gather_ಅಡ್ಡಸಾಲುಗಳು_compute_f32_vgn;
    }

    work_ಸರತಿ_func_t ಕೆಲಸಗಾರ_func = (elem_ಗಾತ್ರ == 2) ? gather_ಅಡ್ಡಸಾಲುಗಳು_thread_f16 : gather_ಅಡ್ಡಸಾಲುಗಳು_thread_f32;
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ಕೆಲಸಗಾರ_func, &task, octx->n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
