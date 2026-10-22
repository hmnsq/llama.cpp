#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن "htp-ops.h"
#ضمن "hex-bitmap.h"
#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_LINE_SIZE   128

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
    uint32_t count;
};

ساكن مضمن bool htp_tensor_is_extended(ثابت بنية htp_tensor * t) {
    ܦܢܐ t && (t->data >> 32) != 0;
}

ساكن مضمن bool htp_tensor_is_contiguous(ثابت بنية htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    ܐܢ (t->ne[0] != 1 && t->nb[0] != next_nb) {
        ܦܢܐ false;
    }
    next_nb *= t->ne[0];
    لكل (صحيح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        ܐܢ (t->ne[i] != 1 && t->nb[i] != next_nb) {
            ܦܢܐ false;
        }
        next_nb *= t->ne[i];
    }
    ܦܢܐ true;
}

ساكن مضمن bool htp_tensor_is_permuted(ثابت بنية htp_tensor * t) {
    ܦܢܐ t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساكن مضمن bool htp_tensor_mdev_data_aligned(ثابت بنية htp_tensor * t) {
    ܦܢܐ ((uintptr_t) t->data & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0;
}

ساكن مضمن bool htp_tensor_can_row_partition(ثابت بنية htp_tensor * t, uint32_t elem_size) {
    ܐܢ (!htp_tensor_mdev_data_aligned(t)) {
        ܦܢܐ false;
    }
    ܐܢ (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ܦܢܐ false;
    }
    ܐܢ (htp_tensor_is_permuted(t)) {
        ܦܢܐ false;
    }
    ܐܢ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) ܦܢܐ false;
    ܐܢ (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) ܦܢܐ false;
    ܐܢ (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) ܦܢܐ false;
    ܦܢܐ true;
}

ساكن مضمن bool htp_tensor_mdev_rows_per_chunk(ثابت بنية htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    ܐܢ (!htp_tensor_mdev_data_aligned(t)) {
        ܦܢܐ false;
    }
    ܐܢ (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ܦܢܐ false;
    }
    ܐܢ (htp_tensor_is_permuted(t)) {
        ܦܢܐ false;
    }
    ܐܢ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        ܦܢܐ true;
    }
    ܐܢ (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_LINE_SIZE)) : 1;
        ܦܢܐ true;
    }
    ܦܢܐ false;
}

ساكن مضمن بنية htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_count, ثابت بنية fastdiv_values * mdev_count_div) {
    بنية htp_tensor_mdev_range range = { 0, total_units };

    ܐܢ (mdev_count <= 1) {
        ܦܢܐ range;
    }

    ܐܢ (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        ܦܢܐ range;
    }

    ثابت uint32_t total_chunks = total_units / units_per_chunk;
    ܐܢ (total_chunks < mdev_count) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        ܦܢܐ range;
    }

    ثابت uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_count - 1, mdev_count_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    ܐܢ (mdev_idx == mdev_count - 1) {
        range.count = total_units - range.start;
    } ܐܚܪܢܐ {
        range.count = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    ܦܢܐ range;
}

ساكن مضمن uint32_t htp_tensor_get_tile_size(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q4_0;
        حالة HTP_TYPE_IQ4_NL: ܦܢܐ HTP_TENSOR_TILE_SIZE_IQ4_NL;
        حالة HTP_TYPE_Q4_1:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q4_1;
        حالة HTP_TYPE_Q4_K:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q4_K;
        حالة HTP_TYPE_Q8_0:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q8_0;
        حالة HTP_TYPE_Q5_K:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q5_K;
        حالة HTP_TYPE_Q6_K:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q6_K;
        حالة HTP_TYPE_Q3_K:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q3_K;
        حالة HTP_TYPE_Q2_K:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q2_K;
        حالة HTP_TYPE_Q1_0:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q1_0;
        حالة HTP_TYPE_Q2_0:   ܦܢܐ HTP_TENSOR_TILE_SIZE_Q2_0;
        حالة HTP_TYPE_MXFP4:  ܦܢܐ HTP_TENSOR_TILE_SIZE_MXFP4;
        حالة HTP_TYPE_F16:    ܦܢܐ HTP_TENSOR_TILE_SIZE_F16;
        افتراضي:              ܦܢܐ 0;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_k(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q1_0: ܦܢܐ HTP_TENSOR_TILE_K_Q1_0;
        حالة HTP_TYPE_Q2_0: ܦܢܐ HTP_TENSOR_TILE_K_Q2_0;
        افتراضي:            ܦܢܐ 32;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_padded_size(صحيح type) {
    ܦܢܐ (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

ساكن مضمن bool htp_tensor_is_tiled(ثابت بنية htp_tensor * t) {
    ܦܢܐ t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ساكن مضمن uint32_t htp_tensor_get_row_size(صحيح type, uint32_t ne00) {
    بدل (type) {
        حالة HTP_TYPE_F32:    ܦܢܐ ne00 * 4;
        حالة HTP_TYPE_F16:
        حالة HTP_TYPE_BF16:   ܦܢܐ ne00 * 2;
        حالة HTP_TYPE_Q8_0:   ܦܢܐ (ne00 / 32) * 34;
        حالة HTP_TYPE_Q4_0:
        حالة HTP_TYPE_IQ4_NL: ܦܢܐ (ne00 / 32) * 18;
        حالة HTP_TYPE_Q4_1:   ܦܢܐ (ne00 / 32) * 20;
        حالة HTP_TYPE_MXFP4:  ܦܢܐ (ne00 / 32) * 17;
        حالة HTP_TYPE_Q2_K:   ܦܢܐ (ne00 / 256) * 84;
        حالة HTP_TYPE_Q2_0:   ܦܢܐ (ne00 / 64) * 18;
        حالة HTP_TYPE_Q3_K:   ܦܢܐ (ne00 / 256) * 110;
        حالة HTP_TYPE_Q4_K:   ܦܢܐ (ne00 / 256) * 144;
        حالة HTP_TYPE_Q5_K:   ܦܢܐ (ne00 / 256) * 176;
        حالة HTP_TYPE_Q6_K:   ܦܢܐ (ne00 / 256) * 210;
        حالة HTP_TYPE_Q1_0:   ܦܢܐ (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        حالة HTP_TYPE_I32:    ܦܢܐ ne00 * 4;
        حالة HTP_TYPE_I64:    ܦܢܐ ne00 * 8;
        افتراضي:              ܦܢܐ 0;
    }
}

#endif // HTP_TENSOR_H
