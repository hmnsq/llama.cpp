#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#شامل <stdint.h>
#شامل <stdbool.h>
#شامل "htp-ops.h"
#شامل "hex-bitmap.h"
#شامل "hex-common.h"
#شامل "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_سطر_SIZE   128

// Currently HMX requires 32x32 tile formats (do not change)
#define HTP_TENSOR_TILE_N_COLS      32
#define HTP_TENSOR_TILE_N_ROWS      32

#define HTP_TENSOR_TILE_SIZE_F16    2048
#define HTP_TENSOR_TILE_SIZE_Q4_0   576
#define HTP_TENSOR_TILE_SIZE_Q4_1   640
#define HTP_TENSOR_TILE_SIZE_Q8_0   1088
#define HTP_TENSOR_TILE_SIZE_IQ4_NL 576
#define HTP_TENSOR_TILE_SIZE_MXFP4  544
#define HTP_TENSOR_TILE_SIZE_Q4_K   640
#define HTP_TENSOR_TILE_SIZE_Q5_K   768
#define HTP_TENSOR_TILE_SIZE_Q6_K   896
#define HTP_TENSOR_TILE_SIZE_Q3_K   512
#define HTP_TENSOR_TILE_SIZE_Q2_K   336
#define HTP_TENSOR_TILE_K_Q1_0      128
#define HTP_TENSOR_TILE_SIZE_Q1_0   576
#define HTP_TENSOR_TILE_K_Q2_0      64
#define HTP_TENSOR_TILE_SIZE_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

ساختار htp_tensor_mdev_range {
    uint32_t start;
    uint32_t شمار;
};

ایستا درونخطی bool htp_tensor_is_extended(ثابت ساختار htp_tensor * t) {
    بازگشت t && (t->data >> 32) != 0;
}

ایستا درونخطی bool htp_tensor_is_contiguous(ثابت ساختار htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    اگر (t->ne[0] != 1 && t->nb[0] != next_nb) {
        بازگشت false;
    }
    next_nb *= t->ne[0];
    برای (صحیح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        اگر (t->ne[i] != 1 && t->nb[i] != next_nb) {
            بازگشت false;
        }
        next_nb *= t->ne[i];
    }
    بازگشت true;
}

ایستا درونخطی bool htp_tensor_is_permuted(ثابت ساختار htp_tensor * t) {
    بازگشت t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ایستا درونخطی bool htp_tensor_mdev_data_aligned(ثابت ساختار htp_tensor * t) {
    بازگشت ((uintptr_t) t->data & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0;
}

ایستا درونخطی bool htp_tensor_can_row_partition(ثابت ساختار htp_tensor * t, uint32_t elem_size) {
    اگر (!htp_tensor_mdev_data_aligned(t)) {
        بازگشت false;
    }
    اگر (t->ne[0] != 1 && t->nb[0] != elem_size) {
        بازگشت false;
    }
    اگر (htp_tensor_is_permuted(t)) {
        بازگشت false;
    }
    اگر (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) != 0) بازگشت false;
    اگر (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) != 0) بازگشت false;
    اگر (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) != 0) بازگشت false;
    بازگشت true;
}

ایستا درونخطی bool htp_tensor_mdev_rows_per_chunk(ثابت ساختار htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    اگر (!htp_tensor_mdev_data_aligned(t)) {
        بازگشت false;
    }
    اگر (t->ne[0] != 1 && t->nb[0] != elem_size) {
        بازگشت false;
    }
    اگر (htp_tensor_is_permuted(t)) {
        بازگشت false;
    }
    اگر (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        بازگشت true;
    }
    اگر (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_سطر_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_سطر_SIZE)) : 1;
        بازگشت true;
    }
    بازگشت false;
}

ایستا درونخطی ساختار htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_شمار, ثابت ساختار fastdiv_مقادیر * mdev_شمار_div) {
    ساختار htp_tensor_mdev_range range = { 0, total_units };

    اگر (mdev_شمار <= 1) {
        بازگشت range;
    }

    اگر (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.شمار = (mdev_idx == 0) ? total_units : 0;
        بازگشت range;
    }

    ثابت uint32_t total_chunks = total_units / units_per_chunk;
    اگر (total_chunks < mdev_شمار) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.شمار = (mdev_idx == 0) ? total_units : 0;
        بازگشت range;
    }

    ثابت uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_شمار - 1, mdev_شمار_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    اگر (mdev_idx == mdev_شمار - 1) {
        range.شمار = total_units - range.start;
    } وگرنه {
        range.شمار = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    بازگشت range;
}

ایستا درونخطی uint32_t htp_tensor_get_tile_size(صحیح type) {
    انتخاب (type) {
        حالت HTP_TYPE_Q4_0:   بازگشت HTP_TENSOR_TILE_SIZE_Q4_0;
        حالت HTP_TYPE_IQ4_NL: بازگشت HTP_TENSOR_TILE_SIZE_IQ4_NL;
        حالت HTP_TYPE_Q4_1:   بازگشت HTP_TENSOR_TILE_SIZE_Q4_1;
        حالت HTP_TYPE_Q4_K:   بازگشت HTP_TENSOR_TILE_SIZE_Q4_K;
        حالت HTP_TYPE_Q8_0:   بازگشت HTP_TENSOR_TILE_SIZE_Q8_0;
        حالت HTP_TYPE_Q5_K:   بازگشت HTP_TENSOR_TILE_SIZE_Q5_K;
        حالت HTP_TYPE_Q6_K:   بازگشت HTP_TENSOR_TILE_SIZE_Q6_K;
        حالت HTP_TYPE_Q3_K:   بازگشت HTP_TENSOR_TILE_SIZE_Q3_K;
        حالت HTP_TYPE_Q2_K:   بازگشت HTP_TENSOR_TILE_SIZE_Q2_K;
        حالت HTP_TYPE_Q1_0:   بازگشت HTP_TENSOR_TILE_SIZE_Q1_0;
        حالت HTP_TYPE_Q2_0:   بازگشت HTP_TENSOR_TILE_SIZE_Q2_0;
        حالت HTP_TYPE_MXFP4:  بازگشت HTP_TENSOR_TILE_SIZE_MXFP4;
        حالت HTP_TYPE_F16:    بازگشت HTP_TENSOR_TILE_SIZE_F16;
        پیشفرض:              بازگشت 0;
    }
}

ایستا درونخطی uint32_t htp_tensor_get_tile_k(صحیح type) {
    انتخاب (type) {
        حالت HTP_TYPE_Q1_0: بازگشت HTP_TENSOR_TILE_K_Q1_0;
        حالت HTP_TYPE_Q2_0: بازگشت HTP_TENSOR_TILE_K_Q2_0;
        پیشفرض:            بازگشت 32;
    }
}

ایستا درونخطی uint32_t htp_tensor_get_tile_padded_size(صحیح type) {
    بازگشت (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

ایستا درونخطی bool htp_tensor_is_tiled(ثابت ساختار htp_tensor * t) {
    بازگشت t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ایستا درونخطی uint32_t htp_tensor_get_row_size(صحیح type, uint32_t ne00) {
    انتخاب (type) {
        حالت HTP_TYPE_F32:    بازگشت ne00 * 4;
        حالت HTP_TYPE_F16:
        حالت HTP_TYPE_BF16:   بازگشت ne00 * 2;
        حالت HTP_TYPE_Q8_0:   بازگشت (ne00 / 32) * 34;
        حالت HTP_TYPE_Q4_0:
        حالت HTP_TYPE_IQ4_NL: بازگشت (ne00 / 32) * 18;
        حالت HTP_TYPE_Q4_1:   بازگشت (ne00 / 32) * 20;
        حالت HTP_TYPE_MXFP4:  بازگشت (ne00 / 32) * 17;
        حالت HTP_TYPE_Q2_K:   بازگشت (ne00 / 256) * 84;
        حالت HTP_TYPE_Q2_0:   بازگشت (ne00 / 64) * 18;
        حالت HTP_TYPE_Q3_K:   بازگشت (ne00 / 256) * 110;
        حالت HTP_TYPE_Q4_K:   بازگشت (ne00 / 256) * 144;
        حالت HTP_TYPE_Q5_K:   بازگشت (ne00 / 256) * 176;
        حالت HTP_TYPE_Q6_K:   بازگشت (ne00 / 256) * 210;
        حالت HTP_TYPE_Q1_0:   بازگشت (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        حالت HTP_TYPE_I32:    بازگشت ne00 * 4;
        حالت HTP_TYPE_I64:    بازگشت ne00 * 8;
        پیشفرض:              بازگشت 0;
    }
}

#endif // HTP_TENSOR_H
