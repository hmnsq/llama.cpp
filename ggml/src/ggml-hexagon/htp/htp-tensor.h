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

sa_nracanaa htp_tensor_mdev_range {
    uint32_t start;
    uint32_t count;
};

static anta_hstha bool htp_tensor_is_extended(const sa_nracanaa htp_tensor * t) {
    คืน t && (t->data >> 32) != 0;
}

static anta_hstha bool htp_tensor_is_contiguous(const sa_nracanaa htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    ถ้า (t->ne[0] != 1 && t->nb[0] != next_nb) {
        คืน false;
    }
    next_nb *= t->ne[0];
    สำหรับ (จำนวนเต็ม i = 1; i < HTP_OP_MAX_DIMS; i++) {
        ถ้า (t->ne[i] != 1 && t->nb[i] != next_nb) {
            คืน false;
        }
        next_nb *= t->ne[i];
    }
    คืน true;
}

static anta_hstha bool htp_tensor_is_permuted(const sa_nracanaa htp_tensor * t) {
    คืน t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

static anta_hstha bool htp_tensor_mdev_data_aligned(const sa_nracanaa htp_tensor * t) {
    คืน ((uintptr_t) t->data & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0;
}

static anta_hstha bool htp_tensor_can_row_partition(const sa_nracanaa htp_tensor * t, uint32_t elem_size) {
    ถ้า (!htp_tensor_mdev_data_aligned(t)) {
        คืน false;
    }
    ถ้า (t->ne[0] != 1 && t->nb[0] != elem_size) {
        คืน false;
    }
    ถ้า (htp_tensor_is_permuted(t)) {
        คืน false;
    }
    ถ้า (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) คืน false;
    ถ้า (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) คืน false;
    ถ้า (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) คืน false;
    คืน true;
}

static anta_hstha bool htp_tensor_mdev_rows_per_chunk(const sa_nracanaa htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    ถ้า (!htp_tensor_mdev_data_aligned(t)) {
        คืน false;
    }
    ถ้า (t->ne[0] != 1 && t->nb[0] != elem_size) {
        คืน false;
    }
    ถ้า (htp_tensor_is_permuted(t)) {
        คืน false;
    }
    ถ้า (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        คืน true;
    }
    ถ้า (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_LINE_SIZE)) : 1;
        คืน true;
    }
    คืน false;
}

static anta_hstha sa_nracanaa htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_count, const sa_nracanaa fastdiv_values * mdev_count_div) {
    sa_nracanaa htp_tensor_mdev_range range = { 0, total_units };

    ถ้า (mdev_count <= 1) {
        คืน range;
    }

    ถ้า (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        คืน range;
    }

    const uint32_t total_chunks = total_units / units_per_chunk;
    ถ้า (total_chunks < mdev_count) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        คืน range;
    }

    const uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_count - 1, mdev_count_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    ถ้า (mdev_idx == mdev_count - 1) {
        range.count = total_units - range.start;
    } มิฉะนั้น {
        range.count = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    คืน range;
}

static anta_hstha uint32_t htp_tensor_get_tile_size(จำนวนเต็ม type) {
    switch (type) {
        pak_sa HTP_TYPE_Q4_0:   คืน HTP_TENSOR_TILE_SIZE_Q4_0;
        pak_sa HTP_TYPE_IQ4_NL: คืน HTP_TENSOR_TILE_SIZE_IQ4_NL;
        pak_sa HTP_TYPE_Q4_1:   คืน HTP_TENSOR_TILE_SIZE_Q4_1;
        pak_sa HTP_TYPE_Q4_K:   คืน HTP_TENSOR_TILE_SIZE_Q4_K;
        pak_sa HTP_TYPE_Q8_0:   คืน HTP_TENSOR_TILE_SIZE_Q8_0;
        pak_sa HTP_TYPE_Q5_K:   คืน HTP_TENSOR_TILE_SIZE_Q5_K;
        pak_sa HTP_TYPE_Q6_K:   คืน HTP_TENSOR_TILE_SIZE_Q6_K;
        pak_sa HTP_TYPE_Q3_K:   คืน HTP_TENSOR_TILE_SIZE_Q3_K;
        pak_sa HTP_TYPE_Q2_K:   คืน HTP_TENSOR_TILE_SIZE_Q2_K;
        pak_sa HTP_TYPE_Q1_0:   คืน HTP_TENSOR_TILE_SIZE_Q1_0;
        pak_sa HTP_TYPE_Q2_0:   คืน HTP_TENSOR_TILE_SIZE_Q2_0;
        pak_sa HTP_TYPE_MXFP4:  คืน HTP_TENSOR_TILE_SIZE_MXFP4;
        pak_sa HTP_TYPE_F16:    คืน HTP_TENSOR_TILE_SIZE_F16;
        default:              คืน 0;
    }
}

static anta_hstha uint32_t htp_tensor_get_tile_k(จำนวนเต็ม type) {
    switch (type) {
        pak_sa HTP_TYPE_Q1_0: คืน HTP_TENSOR_TILE_K_Q1_0;
        pak_sa HTP_TYPE_Q2_0: คืน HTP_TENSOR_TILE_K_Q2_0;
        default:            คืน 32;
    }
}

static anta_hstha uint32_t htp_tensor_get_tile_padded_size(จำนวนเต็ม type) {
    คืน (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

static anta_hstha bool htp_tensor_is_tiled(const sa_nracanaa htp_tensor * t) {
    คืน t && (t->flags & HTP_TENSOR_TILED) != 0;
}

static anta_hstha uint32_t htp_tensor_get_row_size(จำนวนเต็ม type, uint32_t ne00) {
    switch (type) {
        pak_sa HTP_TYPE_F32:    คืน ne00 * 4;
        pak_sa HTP_TYPE_F16:
        pak_sa HTP_TYPE_BF16:   คืน ne00 * 2;
        pak_sa HTP_TYPE_Q8_0:   คืน (ne00 / 32) * 34;
        pak_sa HTP_TYPE_Q4_0:
        pak_sa HTP_TYPE_IQ4_NL: คืน (ne00 / 32) * 18;
        pak_sa HTP_TYPE_Q4_1:   คืน (ne00 / 32) * 20;
        pak_sa HTP_TYPE_MXFP4:  คืน (ne00 / 32) * 17;
        pak_sa HTP_TYPE_Q2_K:   คืน (ne00 / 256) * 84;
        pak_sa HTP_TYPE_Q2_0:   คืน (ne00 / 64) * 18;
        pak_sa HTP_TYPE_Q3_K:   คืน (ne00 / 256) * 110;
        pak_sa HTP_TYPE_Q4_K:   คืน (ne00 / 256) * 144;
        pak_sa HTP_TYPE_Q5_K:   คืน (ne00 / 256) * 176;
        pak_sa HTP_TYPE_Q6_K:   คืน (ne00 / 256) * 210;
        pak_sa HTP_TYPE_Q1_0:   คืน (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        pak_sa HTP_TYPE_I32:    คืน ne00 * 4;
        pak_sa HTP_TYPE_I64:    คืน ne00 * 8;
        default:              คืน 0;
    }
}

#endif // HTP_TENSOR_H
