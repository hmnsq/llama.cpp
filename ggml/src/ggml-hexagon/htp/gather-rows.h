#यदि_अपरिभाषितम् HTP_GATHER_पङ्क्तिकाः_H
#परिभाषय HTP_GATHER_पङ्क्तिकाः_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>

#यदि_अपरिभाषितम् GGML_MAX_DIMS
#परिभाषय GGML_MAX_DIMS 4
#यदि_अन्तः

संरचना htp_gather_पङ्क्तिकाः_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_आकार;
    uint32_t बहिर्गमनम्_off;
    uint32_t buf_stride;
    uint32_t खण्डिका_2;
    uint32_t योग_पङ्क्तिकाः;
    uint32_t पङ्क्तिकाः_प्रत्येकम्_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

स्थिर अन्तःस्थ bool htp_gather_पङ्क्तिकाः_solve_layout(
    संरचना htp_gather_पङ्क्तिकाः_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_आकार,
    uint32_t योग_पङ्क्तिकाः,
    uint32_t n_तन्तवः,
    uint32_t mdev_गणनम्,
    uint32_t vtcm_आकार) {

    ध्रुव uint32_t max_ne0 = 128 / elem_आकार;
    ध्रुव uint32_t ne = na + nb;
    यदि (ne == 0 || ne > max_ne0) {
        प्रत्यागम false;
    }

    यदि (elem_आकार != 4 && elem_आकार != 2) {
        प्रत्यागम false;
    }

    ध्रुव uint32_t v_lanes = 128 / elem_आकार;
    ध्रुव uint32_t g       = hex_gcd_u32(ne, v_lanes);
    ध्रुव uint32_t rg      = v_lanes / g;
    ध्रुव uint32_t vg      = ne / g;

    ध्रुव uint32_t खण्डिका_2 = (vtcm_आकार / n_तन्तवः) & ~127u;
    ध्रुव uint32_t योग_chunks = योग_पङ्क्तिकाः / rg;
    ध्रुव uint32_t chunks_प्रत्येकम्_dev = (योग_chunks + mdev_गणनम् - 1) / mdev_गणनम्;
    ध्रुव uint32_t max_पङ्क्तिकाः_प्रत्येकम्_dev = (mdev_गणनम् > 1 && योग_chunks >= mdev_गणनम्)
        ? (chunks_प्रत्येकम्_dev * rg + (योग_पङ्क्तिकाः % rg))
        : योग_पङ्क्तिकाः;
    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्((max_पङ्क्तिकाः_प्रत्येकम्_dev + n_तन्तवः - 1) / n_तन्तवः, rg);

    ध्रुव uint32_t fixed = 3 * vg * 128;
    ध्रुव uint32_t प्रत्येकम्_row = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_आकार;
    यदि (खण्डिका_2 <= fixed + 2 * प्रत्येकम्_row * rg) {
        प्रत्यागम false;
    }

    ध्रुव uint32_t avail_प्रत्येकम्_buf = (खण्डिका_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_प्रत्येकम्_buf / प्रत्येकम्_row, पङ्क्तिकाः_प्रत्येकम्_thread);
    यदि (elem_आकार == 2) {
        ध्रुव uint32_t प्रत्येकम्_row_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        यदि (प्रत्येकम्_row_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / प्रत्येकम्_row_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_आकार = 0;
    uint32_t बहिर्गमनम्_off     = 0;
    uint32_t buf_stride  = 0;

    यावत् (nr_max > 0) {
        v1 = hex_आवर्तनम्_ऊर्ध्वम्((nr_max - 1) * a_nb1 + na * elem_आकार, 128);
        यदि (nb > 0) {
            यदि (b_dense) {
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

        बहिर्गमनम्_off    = hex_आवर्तनम्_ऊर्ध्वम्(region_आकार, 128);
        buf_stride = hex_आवर्तनम्_ऊर्ध्वम्(बहिर्गमनम्_off + nr_max * ne * elem_आकार, 128);

        यदि ((elem_आकार != 2 || region_आकार <= 32768) && fixed + 2 * buf_stride <= खण्डिका_2) {
            भङ्ग;
        }

        nr_max -= rg;
    }

    यदि (nr_max == 0) {
        प्रत्यागम false;
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
    kparams->बहिर्गमनम्_off         = बहिर्गमनम्_off;
    kparams->buf_stride      = buf_stride;
    kparams->खण्डिका_2           = खण्डिका_2;
    kparams->योग_पङ्क्तिकाः      = योग_पङ्क्तिकाः;
    kparams->पङ्क्तिकाः_प्रत्येकम्_thread = पङ्क्तिकाः_प्रत्येकम्_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    प्रत्यागम true;
}

#यदि_परिभाषितम् __cplusplus
संरचना ggml_tensor;
स्थिर अन्तःस्थ bool htp_gather_पङ्क्तिकाः_tensor_outer_contiguous(ध्रुव संरचना ggml_tensor * t) {
    size_t अग्र_nb = ((ध्रुव size_t *) t->nb)[1] * ((ध्रुव int64_t *) t->ne)[1];
    प्रति (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS; i++) {
        यदि (((ध्रुव int64_t *) t->ne)[i] != 1 && ((ध्रुव size_t *) t->nb)[i] != अग्र_nb) {
            प्रत्यागम false;
        }
        अग्र_nb *= ((ध्रुव int64_t *) t->ne)[i];
    }
    प्रत्यागम true;
}
#अन्यथा
संरचना htp_ops_context;
संरचना htp_tensor;

पूर्णाङ्क htp_gather_पङ्क्तिकाः(
    संरचना htp_ops_context * octx,
    ध्रुव संरचना htp_tensor * a,
    ध्रुव संरचना htp_tensor * b,
    ध्रुव संरचना htp_gather_पङ्क्तिकाः_params * kparams,
    uint32_t n_तन्तवः);
#यदि_अन्तः

#यदि_अन्तः // HTP_GATHER_ROWS_H
