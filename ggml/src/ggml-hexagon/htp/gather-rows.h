#నిర్వచితం_కాకపోతే HTP_GATHER_అడ్డువరుసలు_H
#నిర్వచించు HTP_GATHER_అడ్డువరుసలు_H

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdbool.h>

#నిర్వచితం_కాకపోతే GGML_MAX_DIMS
#నిర్వచించు GGML_MAX_DIMS 4
#అయితే_ముగింపు

నిర్మాణం htp_gather_అడ్డువరుసలు_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_కొలత;
    uint32_t బహిర్గతం_off;
    uint32_t buf_stride;
    uint32_t ముక్క_2;
    uint32_t మొత్తం_అడ్డువరుసలు;
    uint32_t అడ్డువరుసలు_చొప్పున_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

స్థిర అంతఃవరుస bool htp_gather_అడ్డువరుసలు_solve_layout(
    నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_కొలత,
    uint32_t మొత్తం_అడ్డువరుసలు,
    uint32_t n_దారాలు,
    uint32_t mdev_లెక్క,
    uint32_t vtcm_కొలత) {

    స్థిరరాశి uint32_t max_ne0 = 128 / elem_కొలత;
    స్థిరరాశి uint32_t ne = na + nb;
    ఒకవేళ (ne == 0 || ne > max_ne0) {
        తిరిగి false;
    }

    ఒకవేళ (elem_కొలత != 4 && elem_కొలత != 2) {
        తిరిగి false;
    }

    స్థిరరాశి uint32_t v_lanes = 128 / elem_కొలత;
    స్థిరరాశి uint32_t g       = hex_గరిష్ఠ_సామాన్య_భాజకం_u32(ne, v_lanes);
    స్థిరరాశి uint32_t rg      = v_lanes / g;
    స్థిరరాశి uint32_t vg      = ne / g;

    స్థిరరాశి uint32_t ముక్క_2 = (vtcm_కొలత / n_దారాలు) & ~127u;
    స్థిరరాశి uint32_t మొత్తం_chunks = మొత్తం_అడ్డువరుసలు / rg;
    స్థిరరాశి uint32_t chunks_చొప్పున_dev = (మొత్తం_chunks + mdev_లెక్క - 1) / mdev_లెక్క;
    స్థిరరాశి uint32_t max_అడ్డువరుసలు_చొప్పున_dev = (mdev_లెక్క > 1 && మొత్తం_chunks >= mdev_లెక్క)
        ? (chunks_చొప్పున_dev * rg + (మొత్తం_అడ్డువరుసలు % rg))
        : మొత్తం_అడ్డువరుసలు;
    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_thread = hex_విడత_పైకి((max_అడ్డువరుసలు_చొప్పున_dev + n_దారాలు - 1) / n_దారాలు, rg);

    స్థిరరాశి uint32_t fixed = 3 * vg * 128;
    స్థిరరాశి uint32_t చొప్పున_అడ్డువరుస = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_కొలత;
    ఒకవేళ (ముక్క_2 <= fixed + 2 * చొప్పున_అడ్డువరుస * rg) {
        తిరిగి false;
    }

    స్థిరరాశి uint32_t avail_చొప్పున_buf = (ముక్క_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_చొప్పున_buf / చొప్పున_అడ్డువరుస, అడ్డువరుసలు_చొప్పున_thread);
    ఒకవేళ (elem_కొలత == 2) {
        స్థిరరాశి uint32_t చొప్పున_అడ్డువరుస_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        ఒకవేళ (చొప్పున_అడ్డువరుస_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / చొప్పున_అడ్డువరుస_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_కొలత = 0;
    uint32_t బహిర్గతం_off     = 0;
    uint32_t buf_stride  = 0;

    అయితే (nr_max > 0) {
        v1 = hex_విడత_పైకి((nr_max - 1) * a_nb1 + na * elem_కొలత, 128);
        ఒకవేళ (nb > 0) {
            ఒకవేళ (b_dense) {
                span1       = elem_కొలత;
                region_కొలత = v1 + (nr_max - 1) * b_nb1 + nb * elem_కొలత;
            } లేకపోతే {
                span1       = (nr_max - 1) * b_nb1 + elem_కొలత;
                region_కొలత = v1 + nb * span1;
            }
        } లేకపోతే {
            span1       = 0;
            region_కొలత = v1;
        }

        బహిర్గతం_off    = hex_విడత_పైకి(region_కొలత, 128);
        buf_stride = hex_విడత_పైకి(బహిర్గతం_off + nr_max * ne * elem_కొలత, 128);

        ఒకవేళ ((elem_కొలత != 2 || region_కొలత <= 32768) && fixed + 2 * buf_stride <= ముక్క_2) {
            ఆపు;
        }

        nr_max -= rg;
    }

    ఒకవేళ (nr_max == 0) {
        తిరిగి false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_కొలత     = region_కొలత;
    kparams->బహిర్గతం_off         = బహిర్గతం_off;
    kparams->buf_stride      = buf_stride;
    kparams->ముక్క_2           = ముక్క_2;
    kparams->మొత్తం_అడ్డువరుసలు      = మొత్తం_అడ్డువరుసలు;
    kparams->అడ్డువరుసలు_చొప్పున_thread = అడ్డువరుసలు_చొప్పున_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    తిరిగి true;
}

#నిర్వచితమైతే __cplusplus
నిర్మాణం ggml_tensor;
స్థిర అంతఃవరుస bool htp_gather_అడ్డువరుసలు_tensor_outer_contiguous(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    size_t తదుపరి_nb = ((స్థిరరాశి size_t *) t->nb)[1] * ((స్థిరరాశి int64_t *) t->ne)[1];
    కొరకు (పూర్ణాంకం i = 2; i < GGML_MAX_DIMS; i++) {
        ఒకవేళ (((స్థిరరాశి int64_t *) t->ne)[i] != 1 && ((స్థిరరాశి size_t *) t->nb)[i] != తదుపరి_nb) {
            తిరిగి false;
        }
        తదుపరి_nb *= ((స్థిరరాశి int64_t *) t->ne)[i];
    }
    తిరిగి true;
}
#లేకపోతే
నిర్మాణం htp_ops_context;
నిర్మాణం htp_tensor;

పూర్ణాంకం htp_gather_అడ్డువరుసలు(
    నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం htp_tensor * a,
    స్థిరరాశి నిర్మాణం htp_tensor * b,
    స్థిరరాశి నిర్మాణం htp_gather_అడ్డువరుసలు_params * kparams,
    uint32_t n_దారాలు);
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_GATHER_ROWS_H
