#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_GATHER_ಅಡ್ಡಸಾಲುಗಳು_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_GATHER_ಅಡ್ಡಸಾಲುಗಳು_H

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_MAX_DIMS
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_MAX_DIMS 4
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_ಗಾತ್ರ;
    uint32_t ಹೊರಹೋಗು_off;
    uint32_t buf_stride;
    uint32_t ತುಂಡು_2;
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_gather_ಅಡ್ಡಸಾಲುಗಳು_solve_layout(
    ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_ಗಾತ್ರ,
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t n_ಎಳೆಗಳು,
    uint32_t mdev_ಎಣಿಕೆ,
    uint32_t vtcm_ಗಾತ್ರ) {

    ಸ್ಥಿರಾಂಕ uint32_t max_ne0 = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t ne = na + nb;
    ಆದರೆ (ne == 0 || ne > max_ne0) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಆದರೆ (elem_ಗಾತ್ರ != 4 && elem_ಗಾತ್ರ != 2) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಸ್ಥಿರಾಂಕ uint32_t v_lanes = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t g       = hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(ne, v_lanes);
    ಸ್ಥಿರಾಂಕ uint32_t rg      = v_lanes / g;
    ಸ್ಥಿರಾಂಕ uint32_t vg      = ne / g;

    ಸ್ಥಿರಾಂಕ uint32_t ತುಂಡು_2 = (vtcm_ಗಾತ್ರ / n_ಎಳೆಗಳು) & ~127u;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_chunks = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು / rg;
    ಸ್ಥಿರಾಂಕ uint32_t chunks_ಪ್ರತಿ_dev = (ಒಟ್ಟು_chunks + mdev_ಎಣಿಕೆ - 1) / mdev_ಎಣಿಕೆ;
    ಸ್ಥಿರಾಂಕ uint32_t max_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_dev = (mdev_ಎಣಿಕೆ > 1 && ಒಟ್ಟು_chunks >= mdev_ಎಣಿಕೆ)
        ? (chunks_ಪ್ರತಿ_dev * rg + (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು % rg))
        : ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ((max_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_dev + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು, rg);

    ಸ್ಥಿರಾಂಕ uint32_t fixed = 3 * vg * 128;
    ಸ್ಥಿರಾಂಕ uint32_t ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_ಗಾತ್ರ;
    ಆದರೆ (ತುಂಡು_2 <= fixed + 2 * ಪ್ರತಿ_ಅಡ್ಡಸಾಲು * rg) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಸ್ಥಿರಾಂಕ uint32_t avail_ಪ್ರತಿ_buf = (ತುಂಡು_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_ಪ್ರತಿ_buf / ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread);
    ಆದರೆ (elem_ಗಾತ್ರ == 2) {
        ಸ್ಥಿರಾಂಕ uint32_t ಪ್ರತಿ_ಅಡ್ಡಸಾಲು_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        ಆದರೆ (ಪ್ರತಿ_ಅಡ್ಡಸಾಲು_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / ಪ್ರತಿ_ಅಡ್ಡಸಾಲು_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_ಗಾತ್ರ = 0;
    uint32_t ಹೊರಹೋಗು_off     = 0;
    uint32_t buf_stride  = 0;

    ಆಗ (nr_max > 0) {
        v1 = hex_ಸುತ್ತು_ಮೇಲೆ((nr_max - 1) * a_nb1 + na * elem_ಗಾತ್ರ, 128);
        ಆದರೆ (nb > 0) {
            ಆದರೆ (b_dense) {
                span1       = elem_ಗಾತ್ರ;
                region_ಗಾತ್ರ = v1 + (nr_max - 1) * b_nb1 + nb * elem_ಗಾತ್ರ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                span1       = (nr_max - 1) * b_nb1 + elem_ಗಾತ್ರ;
                region_ಗಾತ್ರ = v1 + nb * span1;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            span1       = 0;
            region_ಗಾತ್ರ = v1;
        }

        ಹೊರಹೋಗು_off    = hex_ಸುತ್ತು_ಮೇಲೆ(region_ಗಾತ್ರ, 128);
        buf_stride = hex_ಸುತ್ತು_ಮೇಲೆ(ಹೊರಹೋಗು_off + nr_max * ne * elem_ಗಾತ್ರ, 128);

        ಆದರೆ ((elem_ಗಾತ್ರ != 2 || region_ಗಾತ್ರ <= 32768) && fixed + 2 * buf_stride <= ತುಂಡು_2) {
            ನಿಲ್ಲಿಸು;
        }

        nr_max -= rg;
    }

    ಆದರೆ (nr_max == 0) {
        ಹಿಂತಿರುಗಿ false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_ಗಾತ್ರ     = region_ಗಾತ್ರ;
    kparams->ಹೊರಹೋಗು_off         = ಹೊರಹೋಗು_off;
    kparams->buf_stride      = buf_stride;
    kparams->ತುಂಡು_2           = ತುಂಡು_2;
    kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು      = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    kparams->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    ಹಿಂತಿರುಗಿ true;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ರಚನೆ ggml_tensor;
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_gather_ಅಡ್ಡಸಾಲುಗಳು_tensor_outer_contiguous(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    size_t ಮುಂದಿನ_nb = ((ಸ್ಥಿರಾಂಕ size_t *) t->nb)[1] * ((ಸ್ಥಿರಾಂಕ int64_t *) t->ne)[1];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 2; i < GGML_MAX_DIMS; i++) {
        ಆದರೆ (((ಸ್ಥಿರಾಂಕ int64_t *) t->ne)[i] != 1 && ((ಸ್ಥಿರಾಂಕ size_t *) t->nb)[i] != ಮುಂದಿನ_nb) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಮುಂದಿನ_nb *= ((ಸ್ಥಿರಾಂಕ int64_t *) t->ne)[i];
    }
    ಹಿಂತಿರುಗಿ true;
}
#ಇಲ್ಲದಿದ್ದರೆ
ರಚನೆ htp_ops_context;
ರಚನೆ htp_tensor;

ಪೂರ್ಣಾಂಕ htp_gather_ಅಡ್ಡಸಾಲುಗಳು(
    ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * a,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * b,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params * kparams,
    uint32_t n_ಎಳೆಗಳು);
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_GATHER_ROWS_H
