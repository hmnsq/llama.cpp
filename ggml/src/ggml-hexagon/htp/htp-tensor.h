#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "hex-bitmap.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_வரி_SIZE   128

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

கட்டமைப்பு htp_tensor_mdev_range {
    uint32_t start;
    uint32_t மொத்தம்;
};

நிலையான உள்வரி bool htp_tensor_is_extended(மாறிலி கட்டமைப்பு htp_tensor * t) {
    திரும்பு t && (t->data >> 32) != 0;
}

நிலையான உள்வரி bool htp_tensor_is_contiguous(மாறிலி கட்டமைப்பு htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    எனில் (t->ne[0] != 1 && t->nb[0] != next_nb) {
        திரும்பு false;
    }
    next_nb *= t->ne[0];
    க்கு (முழு i = 1; i < HTP_OP_MAX_DIMS; i++) {
        எனில் (t->ne[i] != 1 && t->nb[i] != next_nb) {
            திரும்பு false;
        }
        next_nb *= t->ne[i];
    }
    திரும்பு true;
}

நிலையான உள்வரி bool htp_tensor_is_permuted(மாறிலி கட்டமைப்பு htp_tensor * t) {
    திரும்பு t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

நிலையான உள்வரி bool htp_tensor_mdev_data_aligned(மாறிலி கட்டமைப்பு htp_tensor * t) {
    திரும்பு ((uintptr_t) t->data & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0;
}

நிலையான உள்வரி bool htp_tensor_can_row_partition(மாறிலி கட்டமைப்பு htp_tensor * t, uint32_t elem_size) {
    எனில் (!htp_tensor_mdev_data_aligned(t)) {
        திரும்பு false;
    }
    எனில் (t->ne[0] != 1 && t->nb[0] != elem_size) {
        திரும்பு false;
    }
    எனில் (htp_tensor_is_permuted(t)) {
        திரும்பு false;
    }
    எனில் (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) != 0) திரும்பு false;
    எனில் (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) != 0) திரும்பு false;
    எனில் (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) != 0) திரும்பு false;
    திரும்பு true;
}

நிலையான உள்வரி bool htp_tensor_mdev_rows_per_chunk(மாறிலி கட்டமைப்பு htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    எனில் (!htp_tensor_mdev_data_aligned(t)) {
        திரும்பு false;
    }
    எனில் (t->ne[0] != 1 && t->nb[0] != elem_size) {
        திரும்பு false;
    }
    எனில் (htp_tensor_is_permuted(t)) {
        திரும்பு false;
    }
    எனில் (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        திரும்பு true;
    }
    எனில் (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_வரி_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_வரி_SIZE)) : 1;
        திரும்பு true;
    }
    திரும்பு false;
}

நிலையான உள்வரி கட்டமைப்பு htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_மொத்தம், மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * mdev_மொத்தம்_div) {
    கட்டமைப்பு htp_tensor_mdev_range range = { 0, total_units };

    எனில் (mdev_மொத்தம் <= 1) {
        திரும்பு range;
    }

    எனில் (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.மொத்தம் = (mdev_idx == 0) ? total_units : 0;
        திரும்பு range;
    }

    மாறிலி uint32_t total_chunks = total_units / units_per_chunk;
    எனில் (total_chunks < mdev_மொத்தம்) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.மொத்தம் = (mdev_idx == 0) ? total_units : 0;
        திரும்பு range;
    }

    மாறிலி uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_மொத்தம் - 1, mdev_மொத்தம்_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    எனில் (mdev_idx == mdev_மொத்தம் - 1) {
        range.மொத்தம் = total_units - range.start;
    } இல்லையெனில் {
        range.மொத்தம் = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    திரும்பு range;
}

நிலையான உள்வரி uint32_t htp_tensor_get_tile_size(முழு type) {
    தேர்வு (type) {
        நிலை HTP_TYPE_Q4_0:   திரும்பு HTP_TENSOR_TILE_SIZE_Q4_0;
        நிலை HTP_TYPE_IQ4_NL: திரும்பு HTP_TENSOR_TILE_SIZE_IQ4_NL;
        நிலை HTP_TYPE_Q4_1:   திரும்பு HTP_TENSOR_TILE_SIZE_Q4_1;
        நிலை HTP_TYPE_Q4_K:   திரும்பு HTP_TENSOR_TILE_SIZE_Q4_K;
        நிலை HTP_TYPE_Q8_0:   திரும்பு HTP_TENSOR_TILE_SIZE_Q8_0;
        நிலை HTP_TYPE_Q5_K:   திரும்பு HTP_TENSOR_TILE_SIZE_Q5_K;
        நிலை HTP_TYPE_Q6_K:   திரும்பு HTP_TENSOR_TILE_SIZE_Q6_K;
        நிலை HTP_TYPE_Q3_K:   திரும்பு HTP_TENSOR_TILE_SIZE_Q3_K;
        நிலை HTP_TYPE_Q2_K:   திரும்பு HTP_TENSOR_TILE_SIZE_Q2_K;
        நிலை HTP_TYPE_Q1_0:   திரும்பு HTP_TENSOR_TILE_SIZE_Q1_0;
        நிலை HTP_TYPE_Q2_0:   திரும்பு HTP_TENSOR_TILE_SIZE_Q2_0;
        நிலை HTP_TYPE_MXFP4:  திரும்பு HTP_TENSOR_TILE_SIZE_MXFP4;
        நிலை HTP_TYPE_F16:    திரும்பு HTP_TENSOR_TILE_SIZE_F16;
        இயல்பு:              திரும்பு 0;
    }
}

நிலையான உள்வரி uint32_t htp_tensor_get_tile_k(முழு type) {
    தேர்வு (type) {
        நிலை HTP_TYPE_Q1_0: திரும்பு HTP_TENSOR_TILE_K_Q1_0;
        நிலை HTP_TYPE_Q2_0: திரும்பு HTP_TENSOR_TILE_K_Q2_0;
        இயல்பு:            திரும்பு 32;
    }
}

நிலையான உள்வரி uint32_t htp_tensor_get_tile_padded_size(முழு type) {
    திரும்பு (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

நிலையான உள்வரி bool htp_tensor_is_tiled(மாறிலி கட்டமைப்பு htp_tensor * t) {
    திரும்பு t && (t->flags & HTP_TENSOR_TILED) != 0;
}

நிலையான உள்வரி uint32_t htp_tensor_get_row_size(முழு type, uint32_t ne00) {
    தேர்வு (type) {
        நிலை HTP_TYPE_F32:    திரும்பு ne00 * 4;
        நிலை HTP_TYPE_F16:
        நிலை HTP_TYPE_BF16:   திரும்பு ne00 * 2;
        நிலை HTP_TYPE_Q8_0:   திரும்பு (ne00 / 32) * 34;
        நிலை HTP_TYPE_Q4_0:
        நிலை HTP_TYPE_IQ4_NL: திரும்பு (ne00 / 32) * 18;
        நிலை HTP_TYPE_Q4_1:   திரும்பு (ne00 / 32) * 20;
        நிலை HTP_TYPE_MXFP4:  திரும்பு (ne00 / 32) * 17;
        நிலை HTP_TYPE_Q2_K:   திரும்பு (ne00 / 256) * 84;
        நிலை HTP_TYPE_Q2_0:   திரும்பு (ne00 / 64) * 18;
        நிலை HTP_TYPE_Q3_K:   திரும்பு (ne00 / 256) * 110;
        நிலை HTP_TYPE_Q4_K:   திரும்பு (ne00 / 256) * 144;
        நிலை HTP_TYPE_Q5_K:   திரும்பு (ne00 / 256) * 176;
        நிலை HTP_TYPE_Q6_K:   திரும்பு (ne00 / 256) * 210;
        நிலை HTP_TYPE_Q1_0:   திரும்பு (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        நிலை HTP_TYPE_I32:    திரும்பு ne00 * 4;
        நிலை HTP_TYPE_I64:    திரும்பு ne00 * 8;
        இயல்பு:              திரும்பு 0;
    }
}

#endif // HTP_TENSOR_H
