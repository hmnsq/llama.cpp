#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "hex-bitmap.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_ಸಾಲು_SIZE   128

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

ಸಂರಚನಾ htp_tensor_mdev_range {
    uint32_t start;
    uint32_t ಎಣಿಕೆ;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_tensor_is_extended(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ t && (t->data >> 32) != 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_tensor_is_contiguous(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    ಯದಿ (t->ne[0] != 1 && t->nb[0] != next_nb) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    next_nb *= t->ne[0];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < HTP_OP_MAX_DIMS; i++) {
        ಯದಿ (t->ne[i] != 1 && t->nb[i] != next_nb) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
        next_nb *= t->ne[i];
    }
    ಪ್ರತ್ಯಾಗಮ true;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_tensor_is_permuted(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_tensor_mdev_data_aligned(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ ((uintptr_t) t->data & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) == 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_tensor_can_row_partition(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t, uint32_t elem_size) {
    ಯದಿ (!htp_tensor_mdev_data_aligned(t)) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (htp_tensor_is_permuted(t)) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) != 0) ಪ್ರತ್ಯಾಗಮ false;
    ಯದಿ (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) != 0) ಪ್ರತ್ಯಾಗಮ false;
    ಯದಿ (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) != 0) ಪ್ರತ್ಯಾಗಮ false;
    ಪ್ರತ್ಯಾಗಮ true;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_tensor_mdev_rows_per_chunk(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    ಯದಿ (!htp_tensor_mdev_data_aligned(t)) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (htp_tensor_is_permuted(t)) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        ಪ್ರತ್ಯಾಗಮ true;
    }
    ಯದಿ (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_ಸಾಲು_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_ಸಾಲು_SIZE)) : 1;
        ಪ್ರತ್ಯಾಗಮ true;
    }
    ಪ್ರತ್ಯಾಗಮ false;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಸಂರಚನಾ htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_ಎಣಿಕೆ, ಧ್ರುವ ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು * mdev_ಎಣಿಕೆ_div) {
    ಸಂರಚನಾ htp_tensor_mdev_range range = { 0, total_units };

    ಯದಿ (mdev_ಎಣಿಕೆ <= 1) {
        ಪ್ರತ್ಯಾಗಮ range;
    }

    ಯದಿ (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.ಎಣಿಕೆ = (mdev_idx == 0) ? total_units : 0;
        ಪ್ರತ್ಯಾಗಮ range;
    }

    ಧ್ರುವ uint32_t total_chunks = total_units / units_per_chunk;
    ಯದಿ (total_chunks < mdev_ಎಣಿಕೆ) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.ಎಣಿಕೆ = (mdev_idx == 0) ? total_units : 0;
        ಪ್ರತ್ಯಾಗಮ range;
    }

    ಧ್ರುವ uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_ಎಣಿಕೆ - 1, mdev_ಎಣಿಕೆ_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    ಯದಿ (mdev_idx == mdev_ಎಣಿಕೆ - 1) {
        range.ಎಣಿಕೆ = total_units - range.start;
    } ಅನ್ಯಥಾ {
        range.ಎಣಿಕೆ = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    ಪ್ರತ್ಯಾಗಮ range;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t htp_tensor_get_tile_size(ಪೂರ್ಣಾಙ್ಕ type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ HTP_TYPE_Q4_0:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q4_0;
        ಪಕ್ಷ HTP_TYPE_IQ4_NL: ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_IQ4_NL;
        ಪಕ್ಷ HTP_TYPE_Q4_1:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q4_1;
        ಪಕ್ಷ HTP_TYPE_Q4_K:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q4_K;
        ಪಕ್ಷ HTP_TYPE_Q8_0:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q8_0;
        ಪಕ್ಷ HTP_TYPE_Q5_K:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q5_K;
        ಪಕ್ಷ HTP_TYPE_Q6_K:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q6_K;
        ಪಕ್ಷ HTP_TYPE_Q3_K:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q3_K;
        ಪಕ್ಷ HTP_TYPE_Q2_K:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q2_K;
        ಪಕ್ಷ HTP_TYPE_Q1_0:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q1_0;
        ಪಕ್ಷ HTP_TYPE_Q2_0:   ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_Q2_0;
        ಪಕ್ಷ HTP_TYPE_MXFP4:  ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_MXFP4;
        ಪಕ್ಷ HTP_TYPE_F16:    ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_SIZE_F16;
        ಸ್ವಭಾವ:              ಪ್ರತ್ಯಾಗಮ 0;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t htp_tensor_get_tile_k(ಪೂರ್ಣಾಙ್ಕ type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ HTP_TYPE_Q1_0: ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_K_Q1_0;
        ಪಕ್ಷ HTP_TYPE_Q2_0: ಪ್ರತ್ಯಾಗಮ HTP_TENSOR_TILE_K_Q2_0;
        ಸ್ವಭಾವ:            ಪ್ರತ್ಯಾಗಮ 32;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t htp_tensor_get_tile_padded_size(ಪೂರ್ಣಾಙ್ಕ type) {
    ಪ್ರತ್ಯಾಗಮ (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_tensor_is_tiled(ಧ್ರುವ ಸಂರಚನಾ htp_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t htp_tensor_get_row_size(ಪೂರ್ಣಾಙ್ಕ type, uint32_t ne00) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ HTP_TYPE_F32:    ಪ್ರತ್ಯಾಗಮ ne00 * 4;
        ಪಕ್ಷ HTP_TYPE_F16:
        ಪಕ್ಷ HTP_TYPE_BF16:   ಪ್ರತ್ಯಾಗಮ ne00 * 2;
        ಪಕ್ಷ HTP_TYPE_Q8_0:   ಪ್ರತ್ಯಾಗಮ (ne00 / 32) * 34;
        ಪಕ್ಷ HTP_TYPE_Q4_0:
        ಪಕ್ಷ HTP_TYPE_IQ4_NL: ಪ್ರತ್ಯಾಗಮ (ne00 / 32) * 18;
        ಪಕ್ಷ HTP_TYPE_Q4_1:   ಪ್ರತ್ಯಾಗಮ (ne00 / 32) * 20;
        ಪಕ್ಷ HTP_TYPE_MXFP4:  ಪ್ರತ್ಯಾಗಮ (ne00 / 32) * 17;
        ಪಕ್ಷ HTP_TYPE_Q2_K:   ಪ್ರತ್ಯಾಗಮ (ne00 / 256) * 84;
        ಪಕ್ಷ HTP_TYPE_Q2_0:   ಪ್ರತ್ಯಾಗಮ (ne00 / 64) * 18;
        ಪಕ್ಷ HTP_TYPE_Q3_K:   ಪ್ರತ್ಯಾಗಮ (ne00 / 256) * 110;
        ಪಕ್ಷ HTP_TYPE_Q4_K:   ಪ್ರತ್ಯಾಗಮ (ne00 / 256) * 144;
        ಪಕ್ಷ HTP_TYPE_Q5_K:   ಪ್ರತ್ಯಾಗಮ (ne00 / 256) * 176;
        ಪಕ್ಷ HTP_TYPE_Q6_K:   ಪ್ರತ್ಯಾಗಮ (ne00 / 256) * 210;
        ಪಕ್ಷ HTP_TYPE_Q1_0:   ಪ್ರತ್ಯಾಗಮ (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        ಪಕ್ಷ HTP_TYPE_I32:    ಪ್ರತ್ಯಾಗಮ ne00 * 4;
        ಪಕ್ಷ HTP_TYPE_I64:    ಪ್ರತ್ಯಾಗಮ ne00 * 8;
        ಸ್ವಭಾವ:              ಪ್ರತ್ಯಾಗಮ 0;
    }
}

#endif // HTP_TENSOR_H
