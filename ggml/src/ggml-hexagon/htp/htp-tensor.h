#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "htp-ops.h"
#include "hex-bitmap.h"
#include "hex-common.h"
#include "hex-fastdiv.h"

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

struct htp_tensor_mdev_range {
    uint32_t start;
    uint32_t count;
};

static inline bool htp_tensor_is_extended(const struct htp_tensor * t) {
    반환 t && (t->data >> 32) != 0;
}

static inline bool htp_tensor_is_contiguous(const struct htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    만약 (t->ne[0] != 1 && t->nb[0] != next_nb) {
        반환 false;
    }
    next_nb *= t->ne[0];
    위해 (정수 i = 1; i < HTP_OP_MAX_DIMS; i++) {
        만약 (t->ne[i] != 1 && t->nb[i] != next_nb) {
            반환 false;
        }
        next_nb *= t->ne[i];
    }
    반환 true;
}

static inline bool htp_tensor_is_permuted(const struct htp_tensor * t) {
    반환 t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

static inline bool htp_tensor_mdev_data_aligned(const struct htp_tensor * t) {
    반환 ((uintptr_t) t->data & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0;
}

static inline bool htp_tensor_can_row_partition(const struct htp_tensor * t, uint32_t elem_size) {
    만약 (!htp_tensor_mdev_data_aligned(t)) {
        반환 false;
    }
    만약 (t->ne[0] != 1 && t->nb[0] != elem_size) {
        반환 false;
    }
    만약 (htp_tensor_is_permuted(t)) {
        반환 false;
    }
    만약 (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) 반환 false;
    만약 (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) 반환 false;
    만약 (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) 반환 false;
    반환 true;
}

static inline bool htp_tensor_mdev_rows_per_chunk(const struct htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    만약 (!htp_tensor_mdev_data_aligned(t)) {
        반환 false;
    }
    만약 (t->ne[0] != 1 && t->nb[0] != elem_size) {
        반환 false;
    }
    만약 (htp_tensor_is_permuted(t)) {
        반환 false;
    }
    만약 (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        반환 true;
    }
    만약 (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_LINE_SIZE)) : 1;
        반환 true;
    }
    반환 false;
}

static inline struct htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_count, const struct fastdiv_values * mdev_count_div) {
    struct htp_tensor_mdev_range range = { 0, total_units };

    만약 (mdev_count <= 1) {
        반환 range;
    }

    만약 (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        반환 range;
    }

    const uint32_t total_chunks = total_units / units_per_chunk;
    만약 (total_chunks < mdev_count) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        반환 range;
    }

    const uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_count - 1, mdev_count_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    만약 (mdev_idx == mdev_count - 1) {
        range.count = total_units - range.start;
    } 아니면 {
        range.count = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    반환 range;
}

static inline uint32_t htp_tensor_get_tile_size(정수 type) {
    switch (type) {
        case HTP_TYPE_Q4_0:   반환 HTP_TENSOR_TILE_SIZE_Q4_0;
        case HTP_TYPE_IQ4_NL: 반환 HTP_TENSOR_TILE_SIZE_IQ4_NL;
        case HTP_TYPE_Q4_1:   반환 HTP_TENSOR_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q4_K:   반환 HTP_TENSOR_TILE_SIZE_Q4_K;
        case HTP_TYPE_Q8_0:   반환 HTP_TENSOR_TILE_SIZE_Q8_0;
        case HTP_TYPE_Q5_K:   반환 HTP_TENSOR_TILE_SIZE_Q5_K;
        case HTP_TYPE_Q6_K:   반환 HTP_TENSOR_TILE_SIZE_Q6_K;
        case HTP_TYPE_Q3_K:   반환 HTP_TENSOR_TILE_SIZE_Q3_K;
        case HTP_TYPE_Q2_K:   반환 HTP_TENSOR_TILE_SIZE_Q2_K;
        case HTP_TYPE_Q1_0:   반환 HTP_TENSOR_TILE_SIZE_Q1_0;
        case HTP_TYPE_Q2_0:   반환 HTP_TENSOR_TILE_SIZE_Q2_0;
        case HTP_TYPE_MXFP4:  반환 HTP_TENSOR_TILE_SIZE_MXFP4;
        case HTP_TYPE_F16:    반환 HTP_TENSOR_TILE_SIZE_F16;
        default:              반환 0;
    }
}

static inline uint32_t htp_tensor_get_tile_k(정수 type) {
    switch (type) {
        case HTP_TYPE_Q1_0: 반환 HTP_TENSOR_TILE_K_Q1_0;
        case HTP_TYPE_Q2_0: 반환 HTP_TENSOR_TILE_K_Q2_0;
        default:            반환 32;
    }
}

static inline uint32_t htp_tensor_get_tile_padded_size(정수 type) {
    반환 (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

static inline bool htp_tensor_is_tiled(const struct htp_tensor * t) {
    반환 t && (t->flags & HTP_TENSOR_TILED) != 0;
}

static inline uint32_t htp_tensor_get_row_size(정수 type, uint32_t ne00) {
    switch (type) {
        case HTP_TYPE_F32:    반환 ne00 * 4;
        case HTP_TYPE_F16:
        case HTP_TYPE_BF16:   반환 ne00 * 2;
        case HTP_TYPE_Q8_0:   반환 (ne00 / 32) * 34;
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL: 반환 (ne00 / 32) * 18;
        case HTP_TYPE_Q4_1:   반환 (ne00 / 32) * 20;
        case HTP_TYPE_MXFP4:  반환 (ne00 / 32) * 17;
        case HTP_TYPE_Q2_K:   반환 (ne00 / 256) * 84;
        case HTP_TYPE_Q2_0:   반환 (ne00 / 64) * 18;
        case HTP_TYPE_Q3_K:   반환 (ne00 / 256) * 110;
        case HTP_TYPE_Q4_K:   반환 (ne00 / 256) * 144;
        case HTP_TYPE_Q5_K:   반환 (ne00 / 256) * 176;
        case HTP_TYPE_Q6_K:   반환 (ne00 / 256) * 210;
        case HTP_TYPE_Q1_0:   반환 (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        case HTP_TYPE_I32:    반환 ne00 * 4;
        case HTP_TYPE_I64:    반환 ne00 * 8;
        default:              반환 0;
    }
}

#endif // HTP_TENSOR_H
