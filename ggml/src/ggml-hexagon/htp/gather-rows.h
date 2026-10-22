#اگر_غیر_تعریف_شدہ HTP_GATHER_صفیں_H
#تعریف_کرو HTP_GATHER_صفیں_H

#شامل "hex-common.h"
#شامل <stdint.h>
#شامل <stdbool.h>

#اگر_غیر_تعریف_شدہ GGML_MAX_DIMS
#تعریف_کرو GGML_MAX_DIMS 4
#اگر_ختم

ساخت htp_gather_صفیں_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_سائز;
    uint32_t برآمد_off;
    uint32_t buf_stride;
    uint32_t قاش_2;
    uint32_t کل_صفیں;
    uint32_t صفیں_فی_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ساکن اندرونی bool htp_gather_صفیں_solve_layout(
    ساخت htp_gather_صفیں_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_سائز,
    uint32_t کل_صفیں,
    uint32_t n_دھاگے,
    uint32_t mdev_گنتی,
    uint32_t vtcm_سائز) {

    ثابت uint32_t max_ne0 = 128 / elem_سائز;
    ثابت uint32_t ne = na + nb;
    اگر (ne == 0 || ne > max_ne0) {
        واپس false;
    }

    اگر (elem_سائز != 4 && elem_سائز != 2) {
        واپس false;
    }

    ثابت uint32_t v_lanes = 128 / elem_سائز;
    ثابت uint32_t g       = hex_عاد_اعظم_u32(ne, v_lanes);
    ثابت uint32_t rg      = v_lanes / g;
    ثابت uint32_t vg      = ne / g;

    ثابت uint32_t قاش_2 = (vtcm_سائز / n_دھاگے) & ~127u;
    ثابت uint32_t کل_chunks = کل_صفیں / rg;
    ثابت uint32_t chunks_فی_dev = (کل_chunks + mdev_گنتی - 1) / mdev_گنتی;
    ثابت uint32_t max_صفیں_فی_dev = (mdev_گنتی > 1 && کل_chunks >= mdev_گنتی)
        ? (chunks_فی_dev * rg + (کل_صفیں % rg))
        : کل_صفیں;
    ثابت uint32_t صفیں_فی_thread = hex_باری_اوپر((max_صفیں_فی_dev + n_دھاگے - 1) / n_دھاگے, rg);

    ثابت uint32_t fixed = 3 * vg * 128;
    ثابت uint32_t فی_صف = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_سائز;
    اگر (قاش_2 <= fixed + 2 * فی_صف * rg) {
        واپس false;
    }

    ثابت uint32_t avail_فی_buf = (قاش_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_فی_buf / فی_صف, صفیں_فی_thread);
    اگر (elem_سائز == 2) {
        ثابت uint32_t فی_صف_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        اگر (فی_صف_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / فی_صف_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_سائز = 0;
    uint32_t برآمد_off     = 0;
    uint32_t buf_stride  = 0;

    جب (nr_max > 0) {
        v1 = hex_باری_اوپر((nr_max - 1) * a_nb1 + na * elem_سائز, 128);
        اگر (nb > 0) {
            اگر (b_dense) {
                span1       = elem_سائز;
                region_سائز = v1 + (nr_max - 1) * b_nb1 + nb * elem_سائز;
            } ورنہ {
                span1       = (nr_max - 1) * b_nb1 + elem_سائز;
                region_سائز = v1 + nb * span1;
            }
        } ورنہ {
            span1       = 0;
            region_سائز = v1;
        }

        برآمد_off    = hex_باری_اوپر(region_سائز, 128);
        buf_stride = hex_باری_اوپر(برآمد_off + nr_max * ne * elem_سائز, 128);

        اگر ((elem_سائز != 2 || region_سائز <= 32768) && fixed + 2 * buf_stride <= قاش_2) {
            توڑو;
        }

        nr_max -= rg;
    }

    اگر (nr_max == 0) {
        واپس false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_سائز     = region_سائز;
    kparams->برآمد_off         = برآمد_off;
    kparams->buf_stride      = buf_stride;
    kparams->قاش_2           = قاش_2;
    kparams->کل_صفیں      = کل_صفیں;
    kparams->صفیں_فی_thread = صفیں_فی_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    واپس true;
}

#اگر_تعریف_شدہ __cplusplus
ساخت ggml_tensor;
ساکن اندرونی bool htp_gather_صفیں_tensor_outer_contiguous(ثابت ساخت ggml_tensor * t) {
    size_t اگلا_nb = ((ثابت size_t *) t->nb)[1] * ((ثابت int64_t *) t->ne)[1];
    برائے (صحیح i = 2; i < GGML_MAX_DIMS; i++) {
        اگر (((ثابت int64_t *) t->ne)[i] != 1 && ((ثابت size_t *) t->nb)[i] != اگلا_nb) {
            واپس false;
        }
        اگلا_nb *= ((ثابت int64_t *) t->ne)[i];
    }
    واپس true;
}
#ورنہ
ساخت htp_ops_context;
ساخت htp_tensor;

صحیح htp_gather_صفیں(
    ساخت htp_ops_context * octx,
    ثابت ساخت htp_tensor * a,
    ثابت ساخت htp_tensor * b,
    ثابت ساخت htp_gather_صفیں_params * kparams,
    uint32_t n_دھاگے);
#اگر_ختم

#اگر_ختم // HTP_GATHER_ROWS_H
