#ifndef HTP_GATHER_ROWS_H
#define HTP_GATHER_ROWS_H

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdbool.h>

#ifndef GGML_MAX_DIMS
#define GGML_MAX_DIMS 4
#endif

কাঠামো htp_gather_rows_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_আকার;
    uint32_t out_off;
    uint32_t buf_stride;
    uint32_t slice;
    uint32_t যোগ_rows;
    uint32_t rows_per_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

স্থির অন্তঃরেখ bool htp_gather_rows_solve_layout(
    কাঠামো htp_gather_rows_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_আকার,
    uint32_t যোগ_rows,
    uint32_t n_threads,
    uint32_t mdev_গুনতি,
    uint32_t vtcm_আকার) {

    ধ্রুবক uint32_t max_ne0 = 128 / elem_আকার;
    ধ্রুবক uint32_t ne = na + nb;
    যদি (ne == 0 || ne > max_ne0) {
        ফেরত false;
    }

    যদি (elem_আকার != 4 && elem_আকার != 2) {
        ফেরত false;
    }

    ধ্রুবক uint32_t v_lanes = 128 / elem_আকার;
    ধ্রুবক uint32_t g       = hex_gcd_u32(ne, v_lanes);
    ধ্রুবক uint32_t rg      = v_lanes / g;
    ধ্রুবক uint32_t vg      = ne / g;

    ধ্রুবক uint32_t slice = (vtcm_আকার / n_threads) & ~127u;
    ধ্রুবক uint32_t যোগ_chunks = যোগ_rows / rg;
    ধ্রুবক uint32_t chunks_per_dev = (যোগ_chunks + mdev_গুনতি - 1) / mdev_গুনতি;
    ধ্রুবক uint32_t max_rows_per_dev = (mdev_গুনতি > 1 && যোগ_chunks >= mdev_গুনতি)
        ? (chunks_per_dev * rg + (যোগ_rows % rg))
        : যোগ_rows;
    ধ্রুবক uint32_t rows_per_thread = hex_round_up((max_rows_per_dev + n_threads - 1) / n_threads, rg);

    ধ্রুবক uint32_t fixed = 3 * vg * 128;
    ধ্রুবক uint32_t per_row = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_আকার;
    যদি (slice <= fixed + 2 * per_row * rg) {
        ফেরত false;
    }

    ধ্রুবক uint32_t avail_per_buf = (slice - fixed) / 2;
    uint32_t nr_max = MIN(avail_per_buf / per_row, rows_per_thread);
    যদি (elem_আকার == 2) {
        ধ্রুবক uint32_t per_row_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        যদি (per_row_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / per_row_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_আকার = 0;
    uint32_t out_off     = 0;
    uint32_t buf_stride  = 0;

    যখন (nr_max > 0) {
        v1 = hex_round_up((nr_max - 1) * a_nb1 + na * elem_আকার, 128);
        যদি (nb > 0) {
            যদি (b_dense) {
                span1       = elem_আকার;
                region_আকার = v1 + (nr_max - 1) * b_nb1 + nb * elem_আকার;
            } অন্যথা {
                span1       = (nr_max - 1) * b_nb1 + elem_আকার;
                region_আকার = v1 + nb * span1;
            }
        } অন্যথা {
            span1       = 0;
            region_আকার = v1;
        }

        out_off    = hex_round_up(region_আকার, 128);
        buf_stride = hex_round_up(out_off + nr_max * ne * elem_আকার, 128);

        যদি ((elem_আকার != 2 || region_আকার <= 32768) && fixed + 2 * buf_stride <= slice) {
            বিরতি;
        }

        nr_max -= rg;
    }

    যদি (nr_max == 0) {
        ফেরত false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_আকার     = region_আকার;
    kparams->out_off         = out_off;
    kparams->buf_stride      = buf_stride;
    kparams->slice           = slice;
    kparams->যোগ_rows      = যোগ_rows;
    kparams->rows_per_thread = rows_per_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    ফেরত true;
}

#ifdef __cplusplus
কাঠামো ggml_tensor;
স্থির অন্তঃরেখ bool htp_gather_rows_tensor_outer_contiguous(ধ্রুবক কাঠামো ggml_tensor * t) {
    size_t পরবর্তী_nb = ((ধ্রুবক size_t *) t->nb)[1] * ((ধ্রুবক int64_t *) t->ne)[1];
    জন্য (পূর্ণসংখ্যা i = 2; i < GGML_MAX_DIMS; i++) {
        যদি (((ধ্রুবক int64_t *) t->ne)[i] != 1 && ((ধ্রুবক size_t *) t->nb)[i] != পরবর্তী_nb) {
            ফেরত false;
        }
        পরবর্তী_nb *= ((ধ্রুবক int64_t *) t->ne)[i];
    }
    ফেরত true;
}
#অন্যথা
কাঠামো htp_ops_context;
কাঠামো htp_tensor;

পূর্ণসংখ্যা htp_gather_rows(
    কাঠামো htp_ops_context * octx,
    ধ্রুবক কাঠামো htp_tensor * a,
    ধ্রুবক কাঠামো htp_tensor * b,
    ধ্রুবক কাঠামো htp_gather_rows_params * kparams,
    uint32_t n_threads);
#endif

#endif // HTP_GATHER_ROWS_H
