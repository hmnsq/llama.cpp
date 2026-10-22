#ifndef HTP_GATHER_ROWS_H
#define HTP_GATHER_ROWS_H

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>

#ifndef GGML_MAX_DIMS
#define GGML_MAX_DIMS 4
#endif

ସଂରଚନା htp_gather_rows_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_size;
    uint32_t out_off;
    uint32_t buf_stride;
    uint32_t slice;
    uint32_t total_rows;
    uint32_t rows_per_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_gather_rows_solve_layout(
    ସଂରଚନା htp_gather_rows_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_size,
    uint32_t total_rows,
    uint32_t n_threads,
    uint32_t mdev_ଗଣନା,
    uint32_t vtcm_size) {

    ଅଚଳ uint32_t max_ne0 = 128 / elem_size;
    ଅଚଳ uint32_t ne = na + nb;
    ଯଦି (ne == 0 || ne > max_ne0) {
        ଫେରନ୍ତୁ false;
    }

    ଯଦି (elem_size != 4 && elem_size != 2) {
        ଫେରନ୍ତୁ false;
    }

    ଅଚଳ uint32_t v_lanes = 128 / elem_size;
    ଅଚଳ uint32_t g       = hex_gcd_u32(ne, v_lanes);
    ଅଚଳ uint32_t rg      = v_lanes / g;
    ଅଚଳ uint32_t vg      = ne / g;

    ଅଚଳ uint32_t slice = (vtcm_size / n_threads) & ~127u;
    ଅଚଳ uint32_t total_chunks = total_rows / rg;
    ଅଚଳ uint32_t chunks_per_dev = (total_chunks + mdev_ଗଣନା - 1) / mdev_ଗଣନା;
    ଅଚଳ uint32_t max_rows_per_dev = (mdev_ଗଣନା > 1 && total_chunks >= mdev_ଗଣନା)
        ? (chunks_per_dev * rg + (total_rows % rg))
        : total_rows;
    ଅଚଳ uint32_t rows_per_thread = hex_round_up((max_rows_per_dev + n_threads - 1) / n_threads, rg);

    ଅଚଳ uint32_t fixed = 3 * vg * 128;
    ଅଚଳ uint32_t per_row = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_size;
    ଯଦି (slice <= fixed + 2 * per_row * rg) {
        ଫେରନ୍ତୁ false;
    }

    ଅଚଳ uint32_t avail_per_buf = (slice - fixed) / 2;
    uint32_t nr_max = MIN(avail_per_buf / per_row, rows_per_thread);
    ଯଦି (elem_size == 2) {
        ଅଚଳ uint32_t per_row_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        ଯଦି (per_row_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / per_row_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_size = 0;
    uint32_t out_off     = 0;
    uint32_t buf_stride  = 0;

    ଯେତେବେଳେ (nr_max > 0) {
        v1 = hex_round_up((nr_max - 1) * a_nb1 + na * elem_size, 128);
        ଯଦି (nb > 0) {
            ଯଦି (b_dense) {
                span1       = elem_size;
                region_size = v1 + (nr_max - 1) * b_nb1 + nb * elem_size;
            } ଅନ୍ୟଥା {
                span1       = (nr_max - 1) * b_nb1 + elem_size;
                region_size = v1 + nb * span1;
            }
        } ଅନ୍ୟଥା {
            span1       = 0;
            region_size = v1;
        }

        out_off    = hex_round_up(region_size, 128);
        buf_stride = hex_round_up(out_off + nr_max * ne * elem_size, 128);

        ଯଦି ((elem_size != 2 || region_size <= 32768) && fixed + 2 * buf_stride <= slice) {
            ଭାଙ୍ଗ;
        }

        nr_max -= rg;
    }

    ଯଦି (nr_max == 0) {
        ଫେରନ୍ତୁ false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_size     = region_size;
    kparams->out_off         = out_off;
    kparams->buf_stride      = buf_stride;
    kparams->slice           = slice;
    kparams->total_rows      = total_rows;
    kparams->rows_per_thread = rows_per_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    ଫେରନ୍ତୁ true;
}

#ifdef __cplusplus
ସଂରଚନା ggml_tensor;
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_gather_rows_tensor_outer_contiguous(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    size_t next_nb = ((ଅଚଳ size_t *) t->nb)[1] * ((ଅଚଳ int64_t *) t->ne)[1];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 2; i < GGML_MAX_DIMS; i++) {
        ଯଦି (((ଅଚଳ int64_t *) t->ne)[i] != 1 && ((ଅଚଳ size_t *) t->nb)[i] != next_nb) {
            ଫେରନ୍ତୁ false;
        }
        next_nb *= ((ଅଚଳ int64_t *) t->ne)[i];
    }
    ଫେରନ୍ତୁ true;
}
#ଅନ୍ୟଥା
ସଂରଚନା htp_ops_context;
ସଂରଚନା htp_tensor;

ପୂର୍ଣ୍ଣସଂଖ୍ୟା htp_gather_rows(
    ସଂରଚନା htp_ops_context * octx,
    ଅଚଳ ସଂରଚନା htp_tensor * a,
    ଅଚଳ ସଂରଚନା htp_tensor * b,
    ଅଚଳ ସଂରଚନା htp_gather_rows_params * kparams,
    uint32_t n_threads);
#endif

#endif // HTP_GATHER_ROWS_H
