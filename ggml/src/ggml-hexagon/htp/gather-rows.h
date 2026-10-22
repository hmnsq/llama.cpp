#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_GATHER_ਪੰਕਤੀਆਂ_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_GATHER_ਪੰਕਤੀਆਂ_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_MAX_DIMS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_MAX_DIMS 4
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_ਮਾਪ;
    uint32_t ਬਾਹਰਲਾ_off;
    uint32_t buf_stride;
    uint32_t ਟੁਕੜਾ_2;
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_gather_ਪੰਕਤੀਆਂ_solve_layout(
    ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_ਮਾਪ,
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ,
    uint32_t n_ਧਾਗੇ,
    uint32_t mdev_ਗਿਣਤੀ,
    uint32_t vtcm_ਮਾਪ) {

    ਅਚਲ uint32_t max_ne0 = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t ne = na + nb;
    ਜੇ (ne == 0 || ne > max_ne0) {
        ਵਾਪਸ false;
    }

    ਜੇ (elem_ਮਾਪ != 4 && elem_ਮਾਪ != 2) {
        ਵਾਪਸ false;
    }

    ਅਚਲ uint32_t v_lanes = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t g       = hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(ne, v_lanes);
    ਅਚਲ uint32_t rg      = v_lanes / g;
    ਅਚਲ uint32_t vg      = ne / g;

    ਅਚਲ uint32_t ਟੁਕੜਾ_2 = (vtcm_ਮਾਪ / n_ਧਾਗੇ) & ~127u;
    ਅਚਲ uint32_t ਕੁੱਲ_chunks = ਕੁੱਲ_ਪੰਕਤੀਆਂ / rg;
    ਅਚਲ uint32_t chunks_ਪ੍ਰਤੀ_dev = (ਕੁੱਲ_chunks + mdev_ਗਿਣਤੀ - 1) / mdev_ਗਿਣਤੀ;
    ਅਚਲ uint32_t max_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_dev = (mdev_ਗਿਣਤੀ > 1 && ਕੁੱਲ_chunks >= mdev_ਗਿਣਤੀ)
        ? (chunks_ਪ੍ਰਤੀ_dev * rg + (ਕੁੱਲ_ਪੰਕਤੀਆਂ % rg))
        : ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ((max_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_dev + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ, rg);

    ਅਚਲ uint32_t fixed = 3 * vg * 128;
    ਅਚਲ uint32_t ਪ੍ਰਤੀ_ਪੰਕਤੀ = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_ਮਾਪ;
    ਜੇ (ਟੁਕੜਾ_2 <= fixed + 2 * ਪ੍ਰਤੀ_ਪੰਕਤੀ * rg) {
        ਵਾਪਸ false;
    }

    ਅਚਲ uint32_t avail_ਪ੍ਰਤੀ_buf = (ਟੁਕੜਾ_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_ਪ੍ਰਤੀ_buf / ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread);
    ਜੇ (elem_ਮਾਪ == 2) {
        ਅਚਲ uint32_t ਪ੍ਰਤੀ_ਪੰਕਤੀ_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        ਜੇ (ਪ੍ਰਤੀ_ਪੰਕਤੀ_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / ਪ੍ਰਤੀ_ਪੰਕਤੀ_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_ਮਾਪ = 0;
    uint32_t ਬਾਹਰਲਾ_off     = 0;
    uint32_t buf_stride  = 0;

    ਜਦੋਂ (nr_max > 0) {
        v1 = hex_ਗੇੜਾ_ਉੱਪਰ((nr_max - 1) * a_nb1 + na * elem_ਮਾਪ, 128);
        ਜੇ (nb > 0) {
            ਜੇ (b_dense) {
                span1       = elem_ਮਾਪ;
                region_ਮਾਪ = v1 + (nr_max - 1) * b_nb1 + nb * elem_ਮਾਪ;
            } ਨਹੀਂਤਾਂ {
                span1       = (nr_max - 1) * b_nb1 + elem_ਮਾਪ;
                region_ਮਾਪ = v1 + nb * span1;
            }
        } ਨਹੀਂਤਾਂ {
            span1       = 0;
            region_ਮਾਪ = v1;
        }

        ਬਾਹਰਲਾ_off    = hex_ਗੇੜਾ_ਉੱਪਰ(region_ਮਾਪ, 128);
        buf_stride = hex_ਗੇੜਾ_ਉੱਪਰ(ਬਾਹਰਲਾ_off + nr_max * ne * elem_ਮਾਪ, 128);

        ਜੇ ((elem_ਮਾਪ != 2 || region_ਮਾਪ <= 32768) && fixed + 2 * buf_stride <= ਟੁਕੜਾ_2) {
            ਰੋਕੋ;
        }

        nr_max -= rg;
    }

    ਜੇ (nr_max == 0) {
        ਵਾਪਸ false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_ਮਾਪ     = region_ਮਾਪ;
    kparams->ਬਾਹਰਲਾ_off         = ਬਾਹਰਲਾ_off;
    kparams->buf_stride      = buf_stride;
    kparams->ਟੁਕੜਾ_2           = ਟੁਕੜਾ_2;
    kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ      = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    kparams->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    ਵਾਪਸ true;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਣਤਰ ggml_tensor;
ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_gather_ਪੰਕਤੀਆਂ_tensor_outer_contiguous(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    size_t ਅਗਲਾ_nb = ((ਅਚਲ size_t *) t->nb)[1] * ((ਅਚਲ int64_t *) t->ne)[1];
    ਲਈ (ਪੂਰਨ i = 2; i < GGML_MAX_DIMS; i++) {
        ਜੇ (((ਅਚਲ int64_t *) t->ne)[i] != 1 && ((ਅਚਲ size_t *) t->nb)[i] != ਅਗਲਾ_nb) {
            ਵਾਪਸ false;
        }
        ਅਗਲਾ_nb *= ((ਅਚਲ int64_t *) t->ne)[i];
    }
    ਵਾਪਸ true;
}
#ਨਹੀਂਤਾਂ
ਬਣਤਰ htp_ops_context;
ਬਣਤਰ htp_tensor;

ਪੂਰਨ htp_gather_ਪੰਕਤੀਆਂ(
    ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ htp_tensor * a,
    ਅਚਲ ਬਣਤਰ htp_tensor * b,
    ਅਚਲ ਬਣਤਰ htp_gather_ਪੰਕਤੀਆਂ_params * kparams,
    uint32_t n_ਧਾਗੇ);
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_GATHER_ROWS_H
