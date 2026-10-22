#که_نه_تعریف_شوی HTP_GATHER_لیکې_H
#تعریف_کړه HTP_GATHER_لیکې_H

#ضمن "hex-common.h"
#ضمن <stdint.h>
#ضمن <stdbool.h>

#که_نه_تعریف_شوی GGML_MAX_DIMS
#تعریف_کړه GGML_MAX_DIMS 4
#د_که_پای

بنية htp_gather_لیکې_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_اندازه;
    uint32_t بهر_off;
    uint32_t buf_stride;
    uint32_t ټوټه_2;
    uint32_t ټول_لیکې;
    uint32_t لیکې_فی_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ساكن مضمن bool htp_gather_لیکې_solve_layout(
    بنية htp_gather_لیکې_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_اندازه,
    uint32_t ټول_لیکې,
    uint32_t n_تارونه,
    uint32_t mdev_شمېر,
    uint32_t vtcm_اندازه) {

    ثابت uint32_t max_ne0 = 128 / elem_اندازه;
    ثابت uint32_t ne = na + nb;
    إذا (ne == 0 || ne > max_ne0) {
        أرجع false;
    }

    إذا (elem_اندازه != 4 && elem_اندازه != 2) {
        أرجع false;
    }

    ثابت uint32_t v_lanes = 128 / elem_اندازه;
    ثابت uint32_t g       = hex_لوی_ګډ_وېشونکی_u32(ne, v_lanes);
    ثابت uint32_t rg      = v_lanes / g;
    ثابت uint32_t vg      = ne / g;

    ثابت uint32_t ټوټه_2 = (vtcm_اندازه / n_تارونه) & ~127u;
    ثابت uint32_t ټول_chunks = ټول_لیکې / rg;
    ثابت uint32_t chunks_فی_dev = (ټول_chunks + mdev_شمېر - 1) / mdev_شمېر;
    ثابت uint32_t max_لیکې_فی_dev = (mdev_شمېر > 1 && ټول_chunks >= mdev_شمېر)
        ? (chunks_فی_dev * rg + (ټول_لیکې % rg))
        : ټول_لیکې;
    ثابت uint32_t لیکې_فی_thread = hex_پړاو_پورته((max_لیکې_فی_dev + n_تارونه - 1) / n_تارونه, rg);

    ثابت uint32_t fixed = 3 * vg * 128;
    ثابت uint32_t فی_لیکه = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_اندازه;
    إذا (ټوټه_2 <= fixed + 2 * فی_لیکه * rg) {
        أرجع false;
    }

    ثابت uint32_t avail_فی_buf = (ټوټه_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_فی_buf / فی_لیکه, لیکې_فی_thread);
    إذا (elem_اندازه == 2) {
        ثابت uint32_t فی_لیکه_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        إذا (فی_لیکه_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / فی_لیکه_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_اندازه = 0;
    uint32_t بهر_off     = 0;
    uint32_t buf_stride  = 0;

    طالما (nr_max > 0) {
        v1 = hex_پړاو_پورته((nr_max - 1) * a_nb1 + na * elem_اندازه, 128);
        إذا (nb > 0) {
            إذا (b_dense) {
                span1       = elem_اندازه;
                region_اندازه = v1 + (nr_max - 1) * b_nb1 + nb * elem_اندازه;
            } وإلا {
                span1       = (nr_max - 1) * b_nb1 + elem_اندازه;
                region_اندازه = v1 + nb * span1;
            }
        } وإلا {
            span1       = 0;
            region_اندازه = v1;
        }

        بهر_off    = hex_پړاو_پورته(region_اندازه, 128);
        buf_stride = hex_پړاو_پورته(بهر_off + nr_max * ne * elem_اندازه, 128);

        إذا ((elem_اندازه != 2 || region_اندازه <= 32768) && fixed + 2 * buf_stride <= ټوټه_2) {
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
    kparams->region_اندازه     = region_اندازه;
    kparams->بهر_off         = بهر_off;
    kparams->buf_stride      = buf_stride;
    kparams->ټوټه_2           = ټوټه_2;
    kparams->ټول_لیکې      = ټول_لیکې;
    kparams->لیکې_فی_thread = لیکې_فی_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    أرجع true;
}

#که_تعریف_شوی __cplusplus
بنية ggml_tensor;
ساكن مضمن bool htp_gather_لیکې_tensor_outer_contiguous(ثابت بنية ggml_tensor * t) {
    size_t بل_nb = ((ثابت size_t *) t->nb)[1] * ((ثابت int64_t *) t->ne)[1];
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        إذا (((ثابت int64_t *) t->ne)[i] != 1 && ((ثابت size_t *) t->nb)[i] != بل_nb) {
            أرجع false;
        }
        بل_nb *= ((ثابت int64_t *) t->ne)[i];
    }
    أرجع true;
}
#وإلا
بنية htp_ops_context;
بنية htp_tensor;

صحيح htp_gather_لیکې(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_لیکې_params * kparams,
    uint32_t n_تارونه);
#د_که_پای

#د_که_پای // HTP_GATHER_ROWS_H
