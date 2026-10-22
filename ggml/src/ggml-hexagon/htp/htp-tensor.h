#اگر_غیر_تعریف_شدہ HTP_TENSOR_H
#تعریف_کرو HTP_TENSOR_H

#شامل <stdint.h>
#شامل <stdbool.h>
#شامل "htp-ops.h"
#شامل "hex-bitmap.h"
#شامل "hex-common.h"
#شامل "hex-fastdiv.h"

#تعریف_کرو HTP_TENSOR_MDEV_سطر_سائز   128

// Currently HMX requires 32x32 tile formats (do not change)
#تعریف_کرو HTP_TENSOR_TILE_N_ستون_تعداد      32
#تعریف_کرو HTP_TENSOR_TILE_N_صفیں      32

#تعریف_کرو HTP_TENSOR_TILE_سائز_F16    2048
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q4_0   576
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q4_1   640
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q8_0   1088
#تعریف_کرو HTP_TENSOR_TILE_سائز_IQ4_NL 576
#تعریف_کرو HTP_TENSOR_TILE_سائز_MXFP4  544
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q4_K   640
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q5_K   768
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q6_K   896
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q3_K   512
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q2_K   336
#تعریف_کرو HTP_TENSOR_TILE_K_Q1_0      128
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q1_0   576
#تعریف_کرو HTP_TENSOR_TILE_K_Q2_0      64
#تعریف_کرو HTP_TENSOR_TILE_سائز_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

ساخت htp_tensor_mdev_range {
    uint32_t آغاز;
    uint32_t گنتی;
};

ساکن اندرونی bool htp_tensor_is_extended(ثابت ساخت htp_tensor * t) {
    واپس t && (t->ڈیٹا >> 32) != 0;
}

ساکن اندرونی bool htp_tensor_is_contiguous(ثابت ساخت htp_tensor * t, uint32_t type_سائز) {
    uint32_t اگلا_nb = type_سائز;
    اگر (t->ne[0] != 1 && t->nb[0] != اگلا_nb) {
        واپس false;
    }
    اگلا_nb *= t->ne[0];
    برائے (صحیح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        اگر (t->ne[i] != 1 && t->nb[i] != اگلا_nb) {
            واپس false;
        }
        اگلا_nb *= t->ne[i];
    }
    واپس true;
}

ساکن اندرونی bool htp_tensor_is_permuted(ثابت ساخت htp_tensor * t) {
    واپس t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساکن اندرونی bool htp_tensor_mdev_ڈیٹا_aligned(ثابت ساخت htp_tensor * t) {
    واپس ((uintptr_t) t->ڈیٹا & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0;
}

ساکن اندرونی bool htp_tensor_can_صف_تقسیم(ثابت ساخت htp_tensor * t, uint32_t elem_سائز) {
    اگر (!htp_tensor_mdev_ڈیٹا_aligned(t)) {
        واپس false;
    }
    اگر (t->ne[0] != 1 && t->nb[0] != elem_سائز) {
        واپس false;
    }
    اگر (htp_tensor_is_permuted(t)) {
        واپس false;
    }
    اگر (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) != 0) واپس false;
    اگر (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) != 0) واپس false;
    اگر (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) != 0) واپس false;
    واپس true;
}

ساکن اندرونی bool htp_tensor_mdev_صفیں_فی_chunk(ثابت ساخت htp_tensor * t, uint32_t elem_سائز, uint32_t صف_سائز, uint32_t * صفیں_فی_chunk) {
    *صفیں_فی_chunk = 0;

    اگر (!htp_tensor_mdev_ڈیٹا_aligned(t)) {
        واپس false;
    }
    اگر (t->ne[0] != 1 && t->nb[0] != elem_سائز) {
        واپس false;
    }
    اگر (htp_tensor_is_permuted(t)) {
        واپس false;
    }
    اگر (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0)) {
        *صفیں_فی_chunk = 1;
        واپس true;
    }
    اگر (t->nb[1] == صف_سائز &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *صفیں_فی_chunk = (صف_سائز > 0) ? (HTP_TENSOR_MDEV_سطر_سائز / hex_عاد_اعظم_u32(صف_سائز, HTP_TENSOR_MDEV_سطر_سائز)) : 1;
        واپس true;
    }
    واپس false;
}

ساکن اندرونی ساخت htp_tensor_mdev_range htp_tensor_mdev_تقسیم(uint32_t کل_units, uint32_t units_فی_chunk, uint32_t mdev_idx, uint32_t mdev_گنتی, ثابت ساخت fastdiv_اقدار * mdev_گنتی_div) {
    ساخت htp_tensor_mdev_range range = { 0, کل_units };

    اگر (mdev_گنتی <= 1) {
        واپس range;
    }

    اگر (units_فی_chunk == 0) {
        range.آغاز = (mdev_idx == 0) ? 0 : کل_units;
        range.گنتی = (mdev_idx == 0) ? کل_units : 0;
        واپس range;
    }

    ثابت uint32_t کل_chunks = کل_units / units_فی_chunk;
    اگر (کل_chunks < mdev_گنتی) {
        range.آغاز = (mdev_idx == 0) ? 0 : کل_units;
        range.گنتی = (mdev_idx == 0) ? کل_units : 0;
        واپس range;
    }

    ثابت uint32_t chunks_فی_mdev = fastdiv(کل_chunks + mdev_گنتی - 1, mdev_گنتی_div);
    range.آغاز = MIN(mdev_idx * chunks_فی_mdev * units_فی_chunk, کل_units);
    اگر (mdev_idx == mdev_گنتی - 1) {
        range.گنتی = کل_units - range.آغاز;
    } ورنہ {
        range.گنتی = MIN(chunks_فی_mdev * units_فی_chunk, کل_units - range.آغاز);
    }
    واپس range;
}

ساکن اندرونی uint32_t htp_tensor_get_tile_سائز(صحیح type) {
    انتخاب (type) {
        حالت HTP_TYPE_Q4_0:   واپس HTP_TENSOR_TILE_سائز_Q4_0;
        حالت HTP_TYPE_IQ4_NL: واپس HTP_TENSOR_TILE_سائز_IQ4_NL;
        حالت HTP_TYPE_Q4_1:   واپس HTP_TENSOR_TILE_سائز_Q4_1;
        حالت HTP_TYPE_Q4_K:   واپس HTP_TENSOR_TILE_سائز_Q4_K;
        حالت HTP_TYPE_Q8_0:   واپس HTP_TENSOR_TILE_سائز_Q8_0;
        حالت HTP_TYPE_Q5_K:   واپس HTP_TENSOR_TILE_سائز_Q5_K;
        حالت HTP_TYPE_Q6_K:   واپس HTP_TENSOR_TILE_سائز_Q6_K;
        حالت HTP_TYPE_Q3_K:   واپس HTP_TENSOR_TILE_سائز_Q3_K;
        حالت HTP_TYPE_Q2_K:   واپس HTP_TENSOR_TILE_سائز_Q2_K;
        حالت HTP_TYPE_Q1_0:   واپس HTP_TENSOR_TILE_سائز_Q1_0;
        حالت HTP_TYPE_Q2_0:   واپس HTP_TENSOR_TILE_سائز_Q2_0;
        حالت HTP_TYPE_MXFP4:  واپس HTP_TENSOR_TILE_سائز_MXFP4;
        حالت HTP_TYPE_F16:    واپس HTP_TENSOR_TILE_سائز_F16;
        طےشدہ:              واپس 0;
    }
}

ساکن اندرونی uint32_t htp_tensor_get_tile_k(صحیح type) {
    انتخاب (type) {
        حالت HTP_TYPE_Q1_0: واپس HTP_TENSOR_TILE_K_Q1_0;
        حالت HTP_TYPE_Q2_0: واپس HTP_TENSOR_TILE_K_Q2_0;
        طےشدہ:            واپس 32;
    }
}

ساکن اندرونی uint32_t htp_tensor_get_tile_padded_سائز(صحیح type) {
    واپس (htp_tensor_get_tile_سائز(type) + 127u) & ~127u;
}

ساکن اندرونی bool htp_tensor_is_tiled(ثابت ساخت htp_tensor * t) {
    واپس t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ساکن اندرونی uint32_t htp_tensor_get_صف_سائز(صحیح type, uint32_t ne00) {
    انتخاب (type) {
        حالت HTP_TYPE_F32:    واپس ne00 * 4;
        حالت HTP_TYPE_F16:
        حالت HTP_TYPE_BF16:   واپس ne00 * 2;
        حالت HTP_TYPE_Q8_0:   واپس (ne00 / 32) * 34;
        حالت HTP_TYPE_Q4_0:
        حالت HTP_TYPE_IQ4_NL: واپس (ne00 / 32) * 18;
        حالت HTP_TYPE_Q4_1:   واپس (ne00 / 32) * 20;
        حالت HTP_TYPE_MXFP4:  واپس (ne00 / 32) * 17;
        حالت HTP_TYPE_Q2_K:   واپس (ne00 / 256) * 84;
        حالت HTP_TYPE_Q2_0:   واپس (ne00 / 64) * 18;
        حالت HTP_TYPE_Q3_K:   واپس (ne00 / 256) * 110;
        حالت HTP_TYPE_Q4_K:   واپس (ne00 / 256) * 144;
        حالت HTP_TYPE_Q5_K:   واپس (ne00 / 256) * 176;
        حالت HTP_TYPE_Q6_K:   واپس (ne00 / 256) * 210;
        حالت HTP_TYPE_Q1_0:   واپس (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        حالت HTP_TYPE_I32:    واپس ne00 * 4;
        حالت HTP_TYPE_I64:    واپس ne00 * 8;
        طےشدہ:              واپس 0;
    }
}

#اگر_ختم // HTP_TENSOR_H
