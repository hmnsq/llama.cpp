#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن "htp-ops.h"
#ضمن "hex-bitmap.h"
#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

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

بنية htp_tensor_mdev_range {
    uint32_t start;
    uint32_t تعداد_2;
};

ساكن مضمن bool htp_tensor_is_extended(ثابت بنية htp_tensor * t) {
    إرجاع t && (t->data >> 32) != 0;
}

ساكن مضمن bool htp_tensor_is_contiguous(ثابت بنية htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    إذا (t->ne[0] != 1 && t->nb[0] != next_nb) {
        إرجاع false;
    }
    next_nb *= t->ne[0];
    لكل (صحيح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        إذا (t->ne[i] != 1 && t->nb[i] != next_nb) {
            إرجاع false;
        }
        next_nb *= t->ne[i];
    }
    إرجاع true;
}

ساكن مضمن bool htp_tensor_is_permuted(ثابت بنية htp_tensor * t) {
    إرجاع t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساكن مضمن bool htp_tensor_mdev_data_aligned(ثابت بنية htp_tensor * t) {
    إرجاع ((uintptr_t) t->data & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0;
}

ساكن مضمن bool htp_tensor_can_row_partition(ثابت بنية htp_tensor * t, uint32_t elem_size) {
    إذا (!htp_tensor_mdev_data_aligned(t)) {
        إرجاع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_size) {
        إرجاع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        إرجاع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) != 0) إرجاع false;
    إذا (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) != 0) إرجاع false;
    إذا (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) != 0) إرجاع false;
    إرجاع true;
}

ساكن مضمن bool htp_tensor_mdev_rows_per_chunk(ثابت بنية htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    إذا (!htp_tensor_mdev_data_aligned(t)) {
        إرجاع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_size) {
        إرجاع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        إرجاع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        إرجاع true;
    }
    إذا (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_سطر_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_سطر_SIZE)) : 1;
        إرجاع true;
    }
    إرجاع false;
}

ساكن مضمن بنية htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_تعداد, ثابت بنية fastdiv_قيم * mdev_تعداد_div) {
    بنية htp_tensor_mdev_range range = { 0, total_units };

    إذا (mdev_تعداد <= 1) {
        إرجاع range;
    }

    إذا (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.تعداد_2 = (mdev_idx == 0) ? total_units : 0;
        إرجاع range;
    }

    ثابت uint32_t total_chunks = total_units / units_per_chunk;
    إذا (total_chunks < mdev_تعداد) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.تعداد_2 = (mdev_idx == 0) ? total_units : 0;
        إرجاع range;
    }

    ثابت uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_تعداد - 1, mdev_تعداد_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    إذا (mdev_idx == mdev_تعداد - 1) {
        range.تعداد_2 = total_units - range.start;
    } وإلا {
        range.تعداد_2 = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    إرجاع range;
}

ساكن مضمن uint32_t htp_tensor_get_tile_size(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:   إرجاع HTP_TENSOR_TILE_SIZE_Q4_0;
        حالة HTP_TYPE_IQ4_NL: إرجاع HTP_TENSOR_TILE_SIZE_IQ4_NL;
        حالة HTP_TYPE_Q4_1:   إرجاع HTP_TENSOR_TILE_SIZE_Q4_1;
        حالة HTP_TYPE_Q4_K:   إرجاع HTP_TENSOR_TILE_SIZE_Q4_K;
        حالة HTP_TYPE_Q8_0:   إرجاع HTP_TENSOR_TILE_SIZE_Q8_0;
        حالة HTP_TYPE_Q5_K:   إرجاع HTP_TENSOR_TILE_SIZE_Q5_K;
        حالة HTP_TYPE_Q6_K:   إرجاع HTP_TENSOR_TILE_SIZE_Q6_K;
        حالة HTP_TYPE_Q3_K:   إرجاع HTP_TENSOR_TILE_SIZE_Q3_K;
        حالة HTP_TYPE_Q2_K:   إرجاع HTP_TENSOR_TILE_SIZE_Q2_K;
        حالة HTP_TYPE_Q1_0:   إرجاع HTP_TENSOR_TILE_SIZE_Q1_0;
        حالة HTP_TYPE_Q2_0:   إرجاع HTP_TENSOR_TILE_SIZE_Q2_0;
        حالة HTP_TYPE_MXFP4:  إرجاع HTP_TENSOR_TILE_SIZE_MXFP4;
        حالة HTP_TYPE_F16:    إرجاع HTP_TENSOR_TILE_SIZE_F16;
        افتراضي:              إرجاع 0;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_k(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q1_0: إرجاع HTP_TENSOR_TILE_K_Q1_0;
        حالة HTP_TYPE_Q2_0: إرجاع HTP_TENSOR_TILE_K_Q2_0;
        افتراضي:            إرجاع 32;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_padded_size(صحيح type) {
    إرجاع (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

ساكن مضمن bool htp_tensor_is_tiled(ثابت بنية htp_tensor * t) {
    إرجاع t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ساكن مضمن uint32_t htp_tensor_get_row_size(صحيح type, uint32_t ne00) {
    بدل (type) {
        حالة HTP_TYPE_F32:    إرجاع ne00 * 4;
        حالة HTP_TYPE_F16:
        حالة HTP_TYPE_BF16:   إرجاع ne00 * 2;
        حالة HTP_TYPE_Q8_0:   إرجاع (ne00 / 32) * 34;
        حالة HTP_TYPE_Q4_0:
        حالة HTP_TYPE_IQ4_NL: إرجاع (ne00 / 32) * 18;
        حالة HTP_TYPE_Q4_1:   إرجاع (ne00 / 32) * 20;
        حالة HTP_TYPE_MXFP4:  إرجاع (ne00 / 32) * 17;
        حالة HTP_TYPE_Q2_K:   إرجاع (ne00 / 256) * 84;
        حالة HTP_TYPE_Q2_0:   إرجاع (ne00 / 64) * 18;
        حالة HTP_TYPE_Q3_K:   إرجاع (ne00 / 256) * 110;
        حالة HTP_TYPE_Q4_K:   إرجاع (ne00 / 256) * 144;
        حالة HTP_TYPE_Q5_K:   إرجاع (ne00 / 256) * 176;
        حالة HTP_TYPE_Q6_K:   إرجاع (ne00 / 256) * 210;
        حالة HTP_TYPE_Q1_0:   إرجاع (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        حالة HTP_TYPE_I32:    إرجاع ne00 * 4;
        حالة HTP_TYPE_I64:    إرجاع ne00 * 8;
        افتراضي:              إرجاع 0;
    }
}

#endif // HTP_TENSOR_H
