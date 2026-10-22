#ifndef HTP_GATHER_ROWS_H
#define HTP_GATHER_ROWS_H

#অন্তর্ভাৱ "hex-common.h"
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <stdbool.h>

#ifndef GGML_MAX_DIMS
#define GGML_MAX_DIMS 4
#endif

সংরচনা htp_gather_rows_params {
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

স্থির অন্তঃস্থ bool htp_gather_rows_solve_layout(
    সংরচনা htp_gather_rows_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_size,
    uint32_t total_rows,
    uint32_t n_threads,
    uint32_t mdev_গুনতি,
    uint32_t vtcm_size) {

    ধ্রুৱ uint32_t max_ne0 = 128 / elem_size;
    ধ্রুৱ uint32_t ne = na + nb;
    যদি (ne == 0 || ne > max_ne0) {
        ঘূৰাই false;
    }

    যদি (elem_size != 4 && elem_size != 2) {
        ঘূৰাই false;
    }

    ধ্রুৱ uint32_t v_lanes = 128 / elem_size;
    ধ্রুৱ uint32_t g       = hex_gcd_u32(ne, v_lanes);
    ধ্রুৱ uint32_t rg      = v_lanes / g;
    ধ্রুৱ uint32_t vg      = ne / g;

    ধ্রুৱ uint32_t slice = (vtcm_size / n_threads) & ~127u;
    ধ্রুৱ uint32_t total_chunks = total_rows / rg;
    ধ্রুৱ uint32_t chunks_per_dev = (total_chunks + mdev_গুনতি - 1) / mdev_গুনতি;
    ধ্রুৱ uint32_t max_rows_per_dev = (mdev_গুনতি > 1 && total_chunks >= mdev_গুনতি)
        ? (chunks_per_dev * rg + (total_rows % rg))
        : total_rows;
    ধ্রুৱ uint32_t rows_per_thread = hex_round_up((max_rows_per_dev + n_threads - 1) / n_threads, rg);

    ধ্রুৱ uint32_t fixed = 3 * vg * 128;
    ধ্রুৱ uint32_t per_row = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_size;
    যদি (slice <= fixed + 2 * per_row * rg) {
        ঘূৰাই false;
    }

    ধ্রুৱ uint32_t avail_per_buf = (slice - fixed) / 2;
    uint32_t nr_max = MIN(avail_per_buf / per_row, rows_per_thread);
    যদি (elem_size == 2) {
        ধ্রুৱ uint32_t per_row_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        যদি (per_row_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / per_row_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_size = 0;
    uint32_t out_off     = 0;
    uint32_t buf_stride  = 0;

    যেতিয়ালৈ (nr_max > 0) {
        v1 = hex_round_up((nr_max - 1) * a_nb1 + na * elem_size, 128);
        যদি (nb > 0) {
            যদি (b_dense) {
                span1       = elem_size;
                region_size = v1 + (nr_max - 1) * b_nb1 + nb * elem_size;
            } অন্যথা {
                span1       = (nr_max - 1) * b_nb1 + elem_size;
                region_size = v1 + nb * span1;
            }
        } অন্যথা {
            span1       = 0;
            region_size = v1;
        }

        out_off    = hex_round_up(region_size, 128);
        buf_stride = hex_round_up(out_off + nr_max * ne * elem_size, 128);

        যদি ((elem_size != 2 || region_size <= 32768) && fixed + 2 * buf_stride <= slice) {
            ভঙ্গ;
        }

        nr_max -= rg;
    }

    যদি (nr_max == 0) {
        ঘূৰাই false;
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
    ঘূৰাই true;
}

#ifdef __cplusplus
সংরচনা ggml_tensor;
স্থির অন্তঃস্থ bool htp_gather_rows_tensor_outer_contiguous(ধ্রুৱ সংরচনা ggml_tensor * t) {
    size_t next_nb = ((ধ্রুৱ size_t *) t->nb)[1] * ((ধ্রুৱ int64_t *) t->ne)[1];
    বাবে (পূর্ণাঙ্ক i = 2; i < GGML_MAX_DIMS; i++) {
        যদি (((ধ্রুৱ int64_t *) t->ne)[i] != 1 && ((ধ্রুৱ size_t *) t->nb)[i] != next_nb) {
            ঘূৰাই false;
        }
        next_nb *= ((ধ্রুৱ int64_t *) t->ne)[i];
    }
    ঘূৰাই true;
}
#অন্যথা
সংরচনা htp_ops_context;
সংরচনা htp_tensor;

পূর্ণাঙ্ক htp_gather_rows(
    সংরচনা htp_ops_context * octx,
    ধ্রুৱ সংরচনা htp_tensor * a,
    ধ্রুৱ সংরচনা htp_tensor * b,
    ধ্রুৱ সংরচনা htp_gather_rows_params * kparams,
    uint32_t n_threads);
#endif

#endif // HTP_GATHER_ROWS_H
