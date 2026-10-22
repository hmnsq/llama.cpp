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
    ආපසු t && (t->data >> 32) != 0;
}

static anta_hstha bool htp_tensor_is_contiguous(const sa_nracanaa htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    නම් (t->ne[0] != 1 && t->nb[0] != next_nb) {
        ආපසු false;
    }
    next_nb *= t->ne[0];
    සඳහා (පූර්ණ_සංඛ්‍යා i = 1; i < HTP_OP_MAX_DIMS; i++) {
        නම් (t->ne[i] != 1 && t->nb[i] != next_nb) {
            ආපසු false;
        }
        next_nb *= t->ne[i];
    }
    ආපසු true;
}

static anta_hstha bool htp_tensor_is_permuted(const sa_nracanaa htp_tensor * t) {
    ආපසු t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

static anta_hstha bool htp_tensor_mdev_data_aligned(const sa_nracanaa htp_tensor * t) {
    ආපසු ((uintptr_t) t->data & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0;
}

static anta_hstha bool htp_tensor_can_row_partition(const sa_nracanaa htp_tensor * t, uint32_t elem_size) {
    නම් (!htp_tensor_mdev_data_aligned(t)) {
        ආපසු false;
    }
    නම් (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ආපසු false;
    }
    නම් (htp_tensor_is_permuted(t)) {
        ආපසු false;
    }
    නම් (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) ආපසු false;
    නම් (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) ආපසු false;
    නම් (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) ආපසු false;
    ආපසු true;
}

static anta_hstha bool htp_tensor_mdev_rows_per_chunk(const sa_nracanaa htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    නම් (!htp_tensor_mdev_data_aligned(t)) {
        ආපසු false;
    }
    නම් (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ආපසු false;
    }
    නම් (htp_tensor_is_permuted(t)) {
        ආපසු false;
    }
    නම් (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        ආපසු true;
    }
    නම් (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_LINE_SIZE)) : 1;
        ආපසු true;
    }
    ආපසු false;
}

static anta_hstha sa_nracanaa htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_count, const sa_nracanaa fastdiv_values * mdev_count_div) {
    sa_nracanaa htp_tensor_mdev_range range = { 0, total_units };

    නම් (mdev_count <= 1) {
        ආපසු range;
    }

    නම් (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        ආපසු range;
    }

    const uint32_t total_chunks = total_units / units_per_chunk;
    නම් (total_chunks < mdev_count) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        ආපසු range;
    }

    const uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_count - 1, mdev_count_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    නම් (mdev_idx == mdev_count - 1) {
        range.count = total_units - range.start;
    } නැතහොත් {
        range.count = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    ආපසු range;
}

static anta_hstha uint32_t htp_tensor_get_tile_size(පූර්ණ_සංඛ්‍යා type) {
    switch (type) {
        pak_sa HTP_TYPE_Q4_0:   ආපසු HTP_TENSOR_TILE_SIZE_Q4_0;
        pak_sa HTP_TYPE_IQ4_NL: ආපසු HTP_TENSOR_TILE_SIZE_IQ4_NL;
        pak_sa HTP_TYPE_Q4_1:   ආපසු HTP_TENSOR_TILE_SIZE_Q4_1;
        pak_sa HTP_TYPE_Q4_K:   ආපසු HTP_TENSOR_TILE_SIZE_Q4_K;
        pak_sa HTP_TYPE_Q8_0:   ආපසු HTP_TENSOR_TILE_SIZE_Q8_0;
        pak_sa HTP_TYPE_Q5_K:   ආපසු HTP_TENSOR_TILE_SIZE_Q5_K;
        pak_sa HTP_TYPE_Q6_K:   ආපසු HTP_TENSOR_TILE_SIZE_Q6_K;
        pak_sa HTP_TYPE_Q3_K:   ආපසු HTP_TENSOR_TILE_SIZE_Q3_K;
        pak_sa HTP_TYPE_Q2_K:   ආපසු HTP_TENSOR_TILE_SIZE_Q2_K;
        pak_sa HTP_TYPE_Q1_0:   ආපසු HTP_TENSOR_TILE_SIZE_Q1_0;
        pak_sa HTP_TYPE_Q2_0:   ආපසු HTP_TENSOR_TILE_SIZE_Q2_0;
        pak_sa HTP_TYPE_MXFP4:  ආපසු HTP_TENSOR_TILE_SIZE_MXFP4;
        pak_sa HTP_TYPE_F16:    ආපසු HTP_TENSOR_TILE_SIZE_F16;
        default:              ආපසු 0;
    }
}

static anta_hstha uint32_t htp_tensor_get_tile_k(පූර්ණ_සංඛ්‍යා type) {
    switch (type) {
        pak_sa HTP_TYPE_Q1_0: ආපසු HTP_TENSOR_TILE_K_Q1_0;
        pak_sa HTP_TYPE_Q2_0: ආපසු HTP_TENSOR_TILE_K_Q2_0;
        default:            ආපසු 32;
    }
}

static anta_hstha uint32_t htp_tensor_get_tile_padded_size(පූර්ණ_සංඛ්‍යා type) {
    ආපසු (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

static anta_hstha bool htp_tensor_is_tiled(const sa_nracanaa htp_tensor * t) {
    ආපසු t && (t->flags & HTP_TENSOR_TILED) != 0;
}

static anta_hstha uint32_t htp_tensor_get_row_size(පූර්ණ_සංඛ්‍යා type, uint32_t ne00) {
    switch (type) {
        pak_sa HTP_TYPE_F32:    ආපසු ne00 * 4;
        pak_sa HTP_TYPE_F16:
        pak_sa HTP_TYPE_BF16:   ආපසු ne00 * 2;
        pak_sa HTP_TYPE_Q8_0:   ආපසු (ne00 / 32) * 34;
        pak_sa HTP_TYPE_Q4_0:
        pak_sa HTP_TYPE_IQ4_NL: ආපසු (ne00 / 32) * 18;
        pak_sa HTP_TYPE_Q4_1:   ආපසු (ne00 / 32) * 20;
        pak_sa HTP_TYPE_MXFP4:  ආපසු (ne00 / 32) * 17;
        pak_sa HTP_TYPE_Q2_K:   ආපසු (ne00 / 256) * 84;
        pak_sa HTP_TYPE_Q2_0:   ආපසු (ne00 / 64) * 18;
        pak_sa HTP_TYPE_Q3_K:   ආපසු (ne00 / 256) * 110;
        pak_sa HTP_TYPE_Q4_K:   ආපසු (ne00 / 256) * 144;
        pak_sa HTP_TYPE_Q5_K:   ආපසු (ne00 / 256) * 176;
        pak_sa HTP_TYPE_Q6_K:   ආපසු (ne00 / 256) * 210;
        pak_sa HTP_TYPE_Q1_0:   ආපසු (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        pak_sa HTP_TYPE_I32:    ආපසු ne00 * 4;
        pak_sa HTP_TYPE_I64:    ආපසු ne00 * 8;
        default:              ආපසු 0;
    }
}

#endif // HTP_TENSOR_H
