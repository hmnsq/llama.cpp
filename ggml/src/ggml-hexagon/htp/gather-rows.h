#جے_بنا_تعریف HTP_GATHER_صفاں_H
#تعریف_کرو HTP_GATHER_صفاں_H

#ضمن "hex-common.h"
#ضمن <stdint.h>
#ضمن <stdbool.h>

#جے_بنا_تعریف GGML_MAX_DIMS
#تعریف_کرو GGML_MAX_DIMS 4
#جے_انت

بنية htp_gather_صفاں_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_ناپ;
    uint32_t باہرلا_off;
    uint32_t buf_stride;
    uint32_t ٹکڑا_2;
    uint32_t کل_صفاں;
    uint32_t صفاں_فی_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ساكن مضمن bool htp_gather_صفاں_solve_layout(
    بنية htp_gather_صفاں_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_ناپ,
    uint32_t کل_صفاں,
    uint32_t n_دھاگے,
    uint32_t mdev_گنتی,
    uint32_t vtcm_ناپ) {

    ثابت uint32_t max_ne0 = 128 / elem_ناپ;
    ثابت uint32_t ne = na + nb;
    إذا (ne == 0 || ne > max_ne0) {
        أرجع false;
    }

    إذا (elem_ناپ != 4 && elem_ناپ != 2) {
        أرجع false;
    }

    ثابت uint32_t v_lanes = 128 / elem_ناپ;
    ثابت uint32_t g       = hex_عاد_اعظم_u32(ne, v_lanes);
    ثابت uint32_t rg      = v_lanes / g;
    ثابت uint32_t vg      = ne / g;

    ثابت uint32_t ٹکڑا_2 = (vtcm_ناپ / n_دھاگے) & ~127u;
    ثابت uint32_t کل_chunks = کل_صفاں / rg;
    ثابت uint32_t chunks_فی_dev = (کل_chunks + mdev_گنتی - 1) / mdev_گنتی;
    ثابت uint32_t max_صفاں_فی_dev = (mdev_گنتی > 1 && کل_chunks >= mdev_گنتی)
        ? (chunks_فی_dev * rg + (کل_صفاں % rg))
        : کل_صفاں;
    ثابت uint32_t صفاں_فی_thread = hex_گیڑا_اتے((max_صفاں_فی_dev + n_دھاگے - 1) / n_دھاگے, rg);

    ثابت uint32_t fixed = 3 * vg * 128;
    ثابت uint32_t فی_صف = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_ناپ;
    إذا (ٹکڑا_2 <= fixed + 2 * فی_صف * rg) {
        أرجع false;
    }

    ثابت uint32_t avail_فی_buf = (ٹکڑا_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_فی_buf / فی_صف, صفاں_فی_thread);
    إذا (elem_ناپ == 2) {
        ثابت uint32_t فی_صف_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        إذا (فی_صف_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / فی_صف_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_ناپ = 0;
    uint32_t باہرلا_off     = 0;
    uint32_t buf_stride  = 0;

    طالما (nr_max > 0) {
        v1 = hex_گیڑا_اتے((nr_max - 1) * a_nb1 + na * elem_ناپ, 128);
        إذا (nb > 0) {
            إذا (b_dense) {
                span1       = elem_ناپ;
                region_ناپ = v1 + (nr_max - 1) * b_nb1 + nb * elem_ناپ;
            } وإلا {
                span1       = (nr_max - 1) * b_nb1 + elem_ناپ;
                region_ناپ = v1 + nb * span1;
            }
        } وإلا {
            span1       = 0;
            region_ناپ = v1;
        }

        باہرلا_off    = hex_گیڑا_اتے(region_ناپ, 128);
        buf_stride = hex_گیڑا_اتے(باہرلا_off + nr_max * ne * elem_ناپ, 128);

        إذا ((elem_ناپ != 2 || region_ناپ <= 32768) && fixed + 2 * buf_stride <= ٹکڑا_2) {
            اكسر;
        }

        nr_max -= rg;
    }

    إذا (nr_max == 0) {
        أرجع false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_ناپ     = region_ناپ;
    kparams->باہرلا_off         = باہرلا_off;
    kparams->buf_stride      = buf_stride;
    kparams->ٹکڑا_2           = ٹکڑا_2;
    kparams->کل_صفاں      = کل_صفاں;
    kparams->صفاں_فی_thread = صفاں_فی_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    أرجع true;
}

#جے_تعریف_شدہ __cplusplus
بنية ggml_tensor;
ساكن مضمن bool htp_gather_صفاں_tensor_outer_contiguous(ثابت بنية ggml_tensor * t) {
    size_t اگلا_nb = ((ثابت size_t *) t->nb)[1] * ((ثابت int64_t *) t->ne)[1];
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        إذا (((ثابت int64_t *) t->ne)[i] != 1 && ((ثابت size_t *) t->nb)[i] != اگلا_nb) {
            أرجع false;
        }
        اگلا_nb *= ((ثابت int64_t *) t->ne)[i];
    }
    أرجع true;
}
#وإلا
بنية htp_ops_context;
بنية htp_tensor;

صحيح htp_gather_صفاں(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_صفاں_params * kparams,
    uint32_t n_دھاگے);
#جے_انت

#جے_انت // HTP_GATHER_ROWS_H
