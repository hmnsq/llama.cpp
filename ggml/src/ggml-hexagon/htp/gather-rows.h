#अगर_अघोषित HTP_GATHER_ROWS_H
#घोषणा HTP_GATHER_ROWS_H

#समावेश "hex-common.h"
#समावेश <stdint.h>
#समावेश <stdbool.h>

#अगर_अघोषित GGML_MAX_DIMS
#घोषणा GGML_MAX_DIMS 4
#पूर्ण_अगर

काष्ठा htp_gather_rows_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_आकार;
    uint32_t out_off;
    uint32_t buf_stride;
    uint32_t slice;
    uint32_t योग_rows;
    uint32_t rows_per_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

अटल अंतरभूत bool htp_gather_rows_solve_layout(
    काष्ठा htp_gather_rows_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_आकार,
    uint32_t योग_rows,
    uint32_t n_threads,
    uint32_t mdev_गिनती,
    uint32_t vtcm_आकार) {

    स्थिर uint32_t max_ne0 = 128 / elem_आकार;
    स्थिर uint32_t ne = na + nb;
    अगर (ne == 0 || ne > max_ne0) {
        वापस false;
    }

    अगर (elem_आकार != 4 && elem_आकार != 2) {
        वापस false;
    }

    स्थिर uint32_t v_lanes = 128 / elem_आकार;
    स्थिर uint32_t g       = hex_gcd_u32(ne, v_lanes);
    स्थिर uint32_t rg      = v_lanes / g;
    स्थिर uint32_t vg      = ne / g;

    स्थिर uint32_t slice = (vtcm_आकार / n_threads) & ~127u;
    स्थिर uint32_t योग_chunks = योग_rows / rg;
    स्थिर uint32_t chunks_per_dev = (योग_chunks + mdev_गिनती - 1) / mdev_गिनती;
    स्थिर uint32_t max_rows_per_dev = (mdev_गिनती > 1 && योग_chunks >= mdev_गिनती)
        ? (chunks_per_dev * rg + (योग_rows % rg))
        : योग_rows;
    स्थिर uint32_t rows_per_thread = hex_round_up((max_rows_per_dev + n_threads - 1) / n_threads, rg);

    स्थिर uint32_t fixed = 3 * vg * 128;
    स्थिर uint32_t per_row = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_आकार;
    अगर (slice <= fixed + 2 * per_row * rg) {
        वापस false;
    }

    स्थिर uint32_t avail_per_buf = (slice - fixed) / 2;
    uint32_t nr_max = MIN(avail_per_buf / per_row, rows_per_thread);
    अगर (elem_आकार == 2) {
        स्थिर uint32_t per_row_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        अगर (per_row_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / per_row_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_आकार = 0;
    uint32_t out_off     = 0;
    uint32_t buf_stride  = 0;

    जबतक (nr_max > 0) {
        v1 = hex_round_up((nr_max - 1) * a_nb1 + na * elem_आकार, 128);
        अगर (nb > 0) {
            अगर (b_dense) {
                span1       = elem_आकार;
                region_आकार = v1 + (nr_max - 1) * b_nb1 + nb * elem_आकार;
            } अन्यथा {
                span1       = (nr_max - 1) * b_nb1 + elem_आकार;
                region_आकार = v1 + nb * span1;
            }
        } अन्यथा {
            span1       = 0;
            region_आकार = v1;
        }

        out_off    = hex_round_up(region_आकार, 128);
        buf_stride = hex_round_up(out_off + nr_max * ne * elem_आकार, 128);

        अगर ((elem_आकार != 2 || region_आकार <= 32768) && fixed + 2 * buf_stride <= slice) {
            अवरोध;
        }

        nr_max -= rg;
    }

    अगर (nr_max == 0) {
        वापस false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_आकार     = region_आकार;
    kparams->out_off         = out_off;
    kparams->buf_stride      = buf_stride;
    kparams->slice           = slice;
    kparams->योग_rows      = योग_rows;
    kparams->rows_per_thread = rows_per_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    वापस true;
}

#अगर_घोषित __cplusplus
काष्ठा ggml_tensor;
अटल अंतरभूत bool htp_gather_rows_tensor_outer_contiguous(स्थिर काष्ठा ggml_tensor * t) {
    माप_प्रकार अग्र_nb = ((स्थिर माप_प्रकार *) t->nb)[1] * ((स्थिर int64_t *) t->ne)[1];
    क्रम (पूर्णांक i = 2; i < GGML_MAX_DIMS; i++) {
        अगर (((स्थिर int64_t *) t->ne)[i] != 1 && ((स्थिर माप_प्रकार *) t->nb)[i] != अग्र_nb) {
            वापस false;
        }
        अग्र_nb *= ((स्थिर int64_t *) t->ne)[i];
    }
    वापस true;
}
#अन्यथा
काष्ठा htp_ops_context;
काष्ठा htp_tensor;

पूर्णांक htp_gather_rows(
    काष्ठा htp_ops_context * octx,
    स्थिर काष्ठा htp_tensor * a,
    स्थिर काष्ठा htp_tensor * b,
    स्थिर काष्ठा htp_gather_rows_params * kparams,
    uint32_t n_threads);
#पूर्ण_अगर

#पूर्ण_अगर // HTP_GATHER_ROWS_H
