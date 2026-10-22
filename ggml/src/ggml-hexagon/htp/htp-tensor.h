#جے_بنا_تعریف HTP_TENSOR_H
#تعریف_کرو HTP_TENSOR_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن "htp-ops.h"
#ضمن "hex-bitmap.h"
#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#تعریف_کرو HTP_TENSOR_MDEV_سطر_ناپ   128

// Currently HMX requires 32x32 tile formats (do not change)
#تعریف_کرو HTP_TENSOR_TILE_N_خانے_گنتی      32
#تعریف_کرو HTP_TENSOR_TILE_N_صفاں      32

#تعریف_کرو HTP_TENSOR_TILE_ناپ_F16    2048
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q4_0   576
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q4_1   640
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q8_0   1088
#تعریف_کرو HTP_TENSOR_TILE_ناپ_IQ4_NL 576
#تعریف_کرو HTP_TENSOR_TILE_ناپ_MXFP4  544
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q4_K   640
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q5_K   768
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q6_K   896
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q3_K   512
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q2_K   336
#تعریف_کرو HTP_TENSOR_TILE_K_Q1_0      128
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q1_0   576
#تعریف_کرو HTP_TENSOR_TILE_K_Q2_0      64
#تعریف_کرو HTP_TENSOR_TILE_ناپ_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

بنية htp_tensor_mdev_range {
    uint32_t شروع;
    uint32_t گنتی;
};

ساكن مضمن bool htp_tensor_is_extended(ثابت بنية htp_tensor * t) {
    أرجع t && (t->انکڑے >> 32) != 0;
}

ساكن مضمن bool htp_tensor_is_contiguous(ثابت بنية htp_tensor * t, uint32_t type_ناپ) {
    uint32_t اگلا_nb = type_ناپ;
    إذا (t->ne[0] != 1 && t->nb[0] != اگلا_nb) {
        أرجع false;
    }
    اگلا_nb *= t->ne[0];
    لكل (صحيح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        إذا (t->ne[i] != 1 && t->nb[i] != اگلا_nb) {
            أرجع false;
        }
        اگلا_nb *= t->ne[i];
    }
    أرجع true;
}

ساكن مضمن bool htp_tensor_is_permuted(ثابت بنية htp_tensor * t) {
    أرجع t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساكن مضمن bool htp_tensor_mdev_انکڑے_aligned(ثابت بنية htp_tensor * t) {
    أرجع ((uintptr_t) t->انکڑے & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) == 0;
}

ساكن مضمن bool htp_tensor_can_صف_حصے_کرو(ثابت بنية htp_tensor * t, uint32_t elem_ناپ) {
    إذا (!htp_tensor_mdev_انکڑے_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_ناپ) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) != 0) أرجع false;
    إذا (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) != 0) أرجع false;
    إذا (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) != 0) أرجع false;
    أرجع true;
}

ساكن مضمن bool htp_tensor_mdev_صفاں_فی_chunk(ثابت بنية htp_tensor * t, uint32_t elem_ناپ, uint32_t صف_ناپ, uint32_t * صفاں_فی_chunk) {
    *صفاں_فی_chunk = 0;

    إذا (!htp_tensor_mdev_انکڑے_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_ناپ) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) == 0)) {
        *صفاں_فی_chunk = 1;
        أرجع true;
    }
    إذا (t->nb[1] == صف_ناپ &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *صفاں_فی_chunk = (صف_ناپ > 0) ? (HTP_TENSOR_MDEV_سطر_ناپ / hex_عاد_اعظم_u32(صف_ناپ, HTP_TENSOR_MDEV_سطر_ناپ)) : 1;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن بنية htp_tensor_mdev_range htp_tensor_mdev_حصے_کرو(uint32_t کل_units, uint32_t units_فی_chunk, uint32_t mdev_idx, uint32_t mdev_گنتی, ثابت بنية fastdiv_اقدار * mdev_گنتی_div) {
    بنية htp_tensor_mdev_range range = { 0, کل_units };

    إذا (mdev_گنتی <= 1) {
        أرجع range;
    }

    إذا (units_فی_chunk == 0) {
        range.شروع = (mdev_idx == 0) ? 0 : کل_units;
        range.گنتی = (mdev_idx == 0) ? کل_units : 0;
        أرجع range;
    }

    ثابت uint32_t کل_chunks = کل_units / units_فی_chunk;
    إذا (کل_chunks < mdev_گنتی) {
        range.شروع = (mdev_idx == 0) ? 0 : کل_units;
        range.گنتی = (mdev_idx == 0) ? کل_units : 0;
        أرجع range;
    }

    ثابت uint32_t chunks_فی_mdev = fastdiv(کل_chunks + mdev_گنتی - 1, mdev_گنتی_div);
    range.شروع = MIN(mdev_idx * chunks_فی_mdev * units_فی_chunk, کل_units);
    إذا (mdev_idx == mdev_گنتی - 1) {
        range.گنتی = کل_units - range.شروع;
    } وإلا {
        range.گنتی = MIN(chunks_فی_mdev * units_فی_chunk, کل_units - range.شروع);
    }
    أرجع range;
}

ساكن مضمن uint32_t htp_tensor_get_tile_ناپ(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:   أرجع HTP_TENSOR_TILE_ناپ_Q4_0;
        حالة HTP_TYPE_IQ4_NL: أرجع HTP_TENSOR_TILE_ناپ_IQ4_NL;
        حالة HTP_TYPE_Q4_1:   أرجع HTP_TENSOR_TILE_ناپ_Q4_1;
        حالة HTP_TYPE_Q4_K:   أرجع HTP_TENSOR_TILE_ناپ_Q4_K;
        حالة HTP_TYPE_Q8_0:   أرجع HTP_TENSOR_TILE_ناپ_Q8_0;
        حالة HTP_TYPE_Q5_K:   أرجع HTP_TENSOR_TILE_ناپ_Q5_K;
        حالة HTP_TYPE_Q6_K:   أرجع HTP_TENSOR_TILE_ناپ_Q6_K;
        حالة HTP_TYPE_Q3_K:   أرجع HTP_TENSOR_TILE_ناپ_Q3_K;
        حالة HTP_TYPE_Q2_K:   أرجع HTP_TENSOR_TILE_ناپ_Q2_K;
        حالة HTP_TYPE_Q1_0:   أرجع HTP_TENSOR_TILE_ناپ_Q1_0;
        حالة HTP_TYPE_Q2_0:   أرجع HTP_TENSOR_TILE_ناپ_Q2_0;
        حالة HTP_TYPE_MXFP4:  أرجع HTP_TENSOR_TILE_ناپ_MXFP4;
        حالة HTP_TYPE_F16:    أرجع HTP_TENSOR_TILE_ناپ_F16;
        افتراضي:              أرجع 0;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_k(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q1_0: أرجع HTP_TENSOR_TILE_K_Q1_0;
        حالة HTP_TYPE_Q2_0: أرجع HTP_TENSOR_TILE_K_Q2_0;
        افتراضي:            أرجع 32;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_padded_ناپ(صحيح type) {
    أرجع (htp_tensor_get_tile_ناپ(type) + 127u) & ~127u;
}

ساكن مضمن bool htp_tensor_is_tiled(ثابت بنية htp_tensor * t) {
    أرجع t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ساكن مضمن uint32_t htp_tensor_get_صف_ناپ(صحيح type, uint32_t ne00) {
    بدل (type) {
        حالة HTP_TYPE_F32:    أرجع ne00 * 4;
        حالة HTP_TYPE_F16:
        حالة HTP_TYPE_BF16:   أرجع ne00 * 2;
        حالة HTP_TYPE_Q8_0:   أرجع (ne00 / 32) * 34;
        حالة HTP_TYPE_Q4_0:
        حالة HTP_TYPE_IQ4_NL: أرجع (ne00 / 32) * 18;
        حالة HTP_TYPE_Q4_1:   أرجع (ne00 / 32) * 20;
        حالة HTP_TYPE_MXFP4:  أرجع (ne00 / 32) * 17;
        حالة HTP_TYPE_Q2_K:   أرجع (ne00 / 256) * 84;
        حالة HTP_TYPE_Q2_0:   أرجع (ne00 / 64) * 18;
        حالة HTP_TYPE_Q3_K:   أرجع (ne00 / 256) * 110;
        حالة HTP_TYPE_Q4_K:   أرجع (ne00 / 256) * 144;
        حالة HTP_TYPE_Q5_K:   أرجع (ne00 / 256) * 176;
        حالة HTP_TYPE_Q6_K:   أرجع (ne00 / 256) * 210;
        حالة HTP_TYPE_Q1_0:   أرجع (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        حالة HTP_TYPE_I32:    أرجع ne00 * 4;
        حالة HTP_TYPE_I64:    أرجع ne00 * 8;
        افتراضي:              أرجع 0;
    }
}

#جے_انت // HTP_TENSOR_H
