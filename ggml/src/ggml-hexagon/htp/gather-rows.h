#اگر_تعریف_نشده HTP_GATHER_ردیفها_H
#تعریف_کن HTP_GATHER_ردیفها_H

#ضمن "hex-common.h"
#ضمن <stdint.h>
#ضمن <stdbool.h>

#اگر_تعریف_نشده GGML_MAX_DIMS
#تعریف_کن GGML_MAX_DIMS 4
#پایان_اگر

بنية htp_gather_ردیفها_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_اندازه;
    uint32_t بیرون_off;
    uint32_t buf_stride;
    uint32_t برش_2;
    uint32_t مجموع_ردیفها;
    uint32_t ردیفها_در_هر_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ساكن مضمن bool htp_gather_ردیفها_solve_layout(
    بنية htp_gather_ردیفها_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_اندازه,
    uint32_t مجموع_ردیفها,
    uint32_t n_نخها,
    uint32_t mdev_شمار,
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
    ثابت uint32_t g       = hex_بزرگترین_مقسوم_علیه_u32(ne, v_lanes);
    ثابت uint32_t rg      = v_lanes / g;
    ثابت uint32_t vg      = ne / g;

    ثابت uint32_t برش_2 = (vtcm_اندازه / n_نخها) & ~127u;
    ثابت uint32_t مجموع_chunks = مجموع_ردیفها / rg;
    ثابت uint32_t chunks_در_هر_dev = (مجموع_chunks + mdev_شمار - 1) / mdev_شمار;
    ثابت uint32_t max_ردیفها_در_هر_dev = (mdev_شمار > 1 && مجموع_chunks >= mdev_شمار)
        ? (chunks_در_هر_dev * rg + (مجموع_ردیفها % rg))
        : مجموع_ردیفها;
    ثابت uint32_t ردیفها_در_هر_thread = hex_دور_به_بالا((max_ردیفها_در_هر_dev + n_نخها - 1) / n_نخها, rg);

    ثابت uint32_t fixed = 3 * vg * 128;
    ثابت uint32_t در_هر_ردیف = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_اندازه;
    إذا (برش_2 <= fixed + 2 * در_هر_ردیف * rg) {
        أرجع false;
    }

    ثابت uint32_t avail_در_هر_buf = (برش_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_در_هر_buf / در_هر_ردیف, ردیفها_در_هر_thread);
    إذا (elem_اندازه == 2) {
        ثابت uint32_t در_هر_ردیف_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        إذا (در_هر_ردیف_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / در_هر_ردیف_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_اندازه = 0;
    uint32_t بیرون_off     = 0;
    uint32_t buf_stride  = 0;

    طالما (nr_max > 0) {
        v1 = hex_دور_به_بالا((nr_max - 1) * a_nb1 + na * elem_اندازه, 128);
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

        بیرون_off    = hex_دور_به_بالا(region_اندازه, 128);
        buf_stride = hex_دور_به_بالا(بیرون_off + nr_max * ne * elem_اندازه, 128);

        إذا ((elem_اندازه != 2 || region_اندازه <= 32768) && fixed + 2 * buf_stride <= برش_2) {
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
    kparams->بیرون_off         = بیرون_off;
    kparams->buf_stride      = buf_stride;
    kparams->برش_2           = برش_2;
    kparams->مجموع_ردیفها      = مجموع_ردیفها;
    kparams->ردیفها_در_هر_thread = ردیفها_در_هر_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    أرجع true;
}

#اگر_تعریف_شده __cplusplus
بنية ggml_tensor;
ساكن مضمن bool htp_gather_ردیفها_tensor_outer_contiguous(ثابت بنية ggml_tensor * t) {
    size_t بعدی_nb = ((ثابت size_t *) t->nb)[1] * ((ثابت int64_t *) t->ne)[1];
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        إذا (((ثابت int64_t *) t->ne)[i] != 1 && ((ثابت size_t *) t->nb)[i] != بعدی_nb) {
            أرجع false;
        }
        بعدی_nb *= ((ثابت int64_t *) t->ne)[i];
    }
    أرجع true;
}
#وإلا
بنية htp_ops_context;
بنية htp_tensor;

صحيح htp_gather_ردیفها(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_ردیفها_params * kparams,
    uint32_t n_نخها);
#پایان_اگر

#پایان_اگر // HTP_GATHER_ROWS_H
