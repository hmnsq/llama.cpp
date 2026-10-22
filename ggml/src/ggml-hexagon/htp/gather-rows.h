#جيڪڏهن_وصف_نه_ٿيل HTP_GATHER_صفون_H
#وصف_ڪريو HTP_GATHER_صفون_H

#ضمن "hex-common.h"
#ضمن <stdint.h>
#ضمن <stdbool.h>

#جيڪڏهن_وصف_نه_ٿيل GGML_MAX_DIMS
#وصف_ڪريو GGML_MAX_DIMS 4
#جيڪڏهن_پورو

بنية htp_gather_صفون_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_سائيز;
    uint32_t ٻاهريون_off;
    uint32_t buf_stride;
    uint32_t ٽڪرو_2;
    uint32_t ڪل_صفون;
    uint32_t صفون_في_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

ساكن مضمن bool htp_gather_صفون_solve_layout(
    بنية htp_gather_صفون_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_سائيز,
    uint32_t ڪل_صفون,
    uint32_t n_ڌاڳا,
    uint32_t mdev_ڳڻپ,
    uint32_t vtcm_سائيز) {

    ثابت uint32_t max_ne0 = 128 / elem_سائيز;
    ثابت uint32_t ne = na + nb;
    إذا (ne == 0 || ne > max_ne0) {
        أرجع false;
    }

    إذا (elem_سائيز != 4 && elem_سائيز != 2) {
        أرجع false;
    }

    ثابت uint32_t v_lanes = 128 / elem_سائيز;
    ثابت uint32_t g       = hex_وڏو_گڏيل_ونڊيندڙ_u32(ne, v_lanes);
    ثابت uint32_t rg      = v_lanes / g;
    ثابت uint32_t vg      = ne / g;

    ثابت uint32_t ٽڪرو_2 = (vtcm_سائيز / n_ڌاڳا) & ~127u;
    ثابت uint32_t ڪل_chunks = ڪل_صفون / rg;
    ثابت uint32_t chunks_في_dev = (ڪل_chunks + mdev_ڳڻپ - 1) / mdev_ڳڻپ;
    ثابت uint32_t max_صفون_في_dev = (mdev_ڳڻپ > 1 && ڪل_chunks >= mdev_ڳڻپ)
        ? (chunks_في_dev * rg + (ڪل_صفون % rg))
        : ڪل_صفون;
    ثابت uint32_t صفون_في_thread = hex_ڦيرو_مٿي((max_صفون_في_dev + n_ڌاڳا - 1) / n_ڌاڳا, rg);

    ثابت uint32_t fixed = 3 * vg * 128;
    ثابت uint32_t في_صف = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_سائيز;
    إذا (ٽڪرو_2 <= fixed + 2 * في_صف * rg) {
        أرجع false;
    }

    ثابت uint32_t avail_في_buf = (ٽڪرو_2 - fixed) / 2;
    uint32_t nr_max = MIN(avail_في_buf / في_صف, صفون_في_thread);
    إذا (elem_سائيز == 2) {
        ثابت uint32_t في_صف_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        إذا (في_صف_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / في_صف_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_سائيز = 0;
    uint32_t ٻاهريون_off     = 0;
    uint32_t buf_stride  = 0;

    طالما (nr_max > 0) {
        v1 = hex_ڦيرو_مٿي((nr_max - 1) * a_nb1 + na * elem_سائيز, 128);
        إذا (nb > 0) {
            إذا (b_dense) {
                span1       = elem_سائيز;
                region_سائيز = v1 + (nr_max - 1) * b_nb1 + nb * elem_سائيز;
            } وإلا {
                span1       = (nr_max - 1) * b_nb1 + elem_سائيز;
                region_سائيز = v1 + nb * span1;
            }
        } وإلا {
            span1       = 0;
            region_سائيز = v1;
        }

        ٻاهريون_off    = hex_ڦيرو_مٿي(region_سائيز, 128);
        buf_stride = hex_ڦيرو_مٿي(ٻاهريون_off + nr_max * ne * elem_سائيز, 128);

        إذا ((elem_سائيز != 2 || region_سائيز <= 32768) && fixed + 2 * buf_stride <= ٽڪرو_2) {
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
    kparams->region_سائيز     = region_سائيز;
    kparams->ٻاهريون_off         = ٻاهريون_off;
    kparams->buf_stride      = buf_stride;
    kparams->ٽڪرو_2           = ٽڪرو_2;
    kparams->ڪل_صفون      = ڪل_صفون;
    kparams->صفون_في_thread = صفون_في_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    أرجع true;
}

#جيڪڏهن_وصف_ٿيل __cplusplus
بنية ggml_tensor;
ساكن مضمن bool htp_gather_صفون_tensor_outer_contiguous(ثابت بنية ggml_tensor * t) {
    size_t ايندڙ_nb = ((ثابت size_t *) t->nb)[1] * ((ثابت int64_t *) t->ne)[1];
    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
        إذا (((ثابت int64_t *) t->ne)[i] != 1 && ((ثابت size_t *) t->nb)[i] != ايندڙ_nb) {
            أرجع false;
        }
        ايندڙ_nb *= ((ثابت int64_t *) t->ne)[i];
    }
    أرجع true;
}
#وإلا
بنية htp_ops_context;
بنية htp_tensor;

صحيح htp_gather_صفون(
    بنية htp_ops_context * octx,
    ثابت بنية htp_tensor * a,
    ثابت بنية htp_tensor * b,
    ثابت بنية htp_gather_صفون_params * kparams,
    uint32_t n_ڌاڳا);
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_GATHER_ROWS_H
