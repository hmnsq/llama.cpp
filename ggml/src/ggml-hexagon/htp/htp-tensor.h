#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "hex-bitmap.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_पङ्क्तिः_SIZE   128

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

संरचना htp_tensor_mdev_range {
    uint32_t start;
    uint32_t गणनम्;
};

स्थिर अन्तःस्थ bool htp_tensor_is_extended(ध्रुव संरचना htp_tensor * t) {
    पटिनिवत्त t && (t->data >> 32) != 0;
}

स्थिर अन्तःस्थ bool htp_tensor_is_contiguous(ध्रुव संरचना htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    सचे (t->ne[0] != 1 && t->nb[0] != next_nb) {
        पटिनिवत्त false;
    }
    next_nb *= t->ne[0];
    अत्थाय (पूर्णाङ्क i = 1; i < HTP_OP_MAX_DIMS; i++) {
        सचे (t->ne[i] != 1 && t->nb[i] != next_nb) {
            पटिनिवत्त false;
        }
        next_nb *= t->ne[i];
    }
    पटिनिवत्त true;
}

स्थिर अन्तःस्थ bool htp_tensor_is_permuted(ध्रुव संरचना htp_tensor * t) {
    पटिनिवत्त t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

स्थिर अन्तःस्थ bool htp_tensor_mdev_data_aligned(ध्रुव संरचना htp_tensor * t) {
    पटिनिवत्त ((uintptr_t) t->data & (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE - 1)) == 0;
}

स्थिर अन्तःस्थ bool htp_tensor_can_row_partition(ध्रुव संरचना htp_tensor * t, uint32_t elem_size) {
    सचे (!htp_tensor_mdev_data_aligned(t)) {
        पटिनिवत्त false;
    }
    सचे (t->ne[0] != 1 && t->nb[0] != elem_size) {
        पटिनिवत्त false;
    }
    सचे (htp_tensor_is_permuted(t)) {
        पटिनिवत्त false;
    }
    सचे (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE - 1)) != 0) पटिनिवत्त false;
    सचे (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE - 1)) != 0) पटिनिवत्त false;
    सचे (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE - 1)) != 0) पटिनिवत्त false;
    पटिनिवत्त true;
}

स्थिर अन्तःस्थ bool htp_tensor_mdev_rows_per_chunk(ध्रुव संरचना htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    सचे (!htp_tensor_mdev_data_aligned(t)) {
        पटिनिवत्त false;
    }
    सचे (t->ne[0] != 1 && t->nb[0] != elem_size) {
        पटिनिवत्त false;
    }
    सचे (htp_tensor_is_permuted(t)) {
        पटिनिवत्त false;
    }
    सचे (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        पटिनिवत्त true;
    }
    सचे (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_पङ्क्तिः_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_पङ्क्तिः_SIZE)) : 1;
        पटिनिवत्त true;
    }
    पटिनिवत्त false;
}

स्थिर अन्तःस्थ संरचना htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_गणनम्, ध्रुव संरचना fastdiv_मानानि * mdev_गणनम्_div) {
    संरचना htp_tensor_mdev_range range = { 0, total_units };

    सचे (mdev_गणनम् <= 1) {
        पटिनिवत्त range;
    }

    सचे (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.गणनम् = (mdev_idx == 0) ? total_units : 0;
        पटिनिवत्त range;
    }

    ध्रुव uint32_t total_chunks = total_units / units_per_chunk;
    सचे (total_chunks < mdev_गणनम्) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.गणनम् = (mdev_idx == 0) ? total_units : 0;
        पटिनिवत्त range;
    }

    ध्रुव uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_गणनम् - 1, mdev_गणनम्_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    सचे (mdev_idx == mdev_गणनम् - 1) {
        range.गणनम् = total_units - range.start;
    } अञ्ञथा {
        range.गणनम् = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    पटिनिवत्त range;
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_tile_size(पूर्णाङ्क type) {
    विकल्प (type) {
        पक्ष HTP_TYPE_Q4_0:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q4_0;
        पक्ष HTP_TYPE_IQ4_NL: पटिनिवत्त HTP_TENSOR_TILE_SIZE_IQ4_NL;
        पक्ष HTP_TYPE_Q4_1:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q4_1;
        पक्ष HTP_TYPE_Q4_K:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q4_K;
        पक्ष HTP_TYPE_Q8_0:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q8_0;
        पक्ष HTP_TYPE_Q5_K:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q5_K;
        पक्ष HTP_TYPE_Q6_K:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q6_K;
        पक्ष HTP_TYPE_Q3_K:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q3_K;
        पक्ष HTP_TYPE_Q2_K:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q2_K;
        पक्ष HTP_TYPE_Q1_0:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q1_0;
        पक्ष HTP_TYPE_Q2_0:   पटिनिवत्त HTP_TENSOR_TILE_SIZE_Q2_0;
        पक्ष HTP_TYPE_MXFP4:  पटिनिवत्त HTP_TENSOR_TILE_SIZE_MXFP4;
        पक्ष HTP_TYPE_F16:    पटिनिवत्त HTP_TENSOR_TILE_SIZE_F16;
        स्वभाव:              पटिनिवत्त 0;
    }
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_tile_k(पूर्णाङ्क type) {
    विकल्प (type) {
        पक्ष HTP_TYPE_Q1_0: पटिनिवत्त HTP_TENSOR_TILE_K_Q1_0;
        पक्ष HTP_TYPE_Q2_0: पटिनिवत्त HTP_TENSOR_TILE_K_Q2_0;
        स्वभाव:            पटिनिवत्त 32;
    }
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_tile_padded_size(पूर्णाङ्क type) {
    पटिनिवत्त (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

स्थिर अन्तःस्थ bool htp_tensor_is_tiled(ध्रुव संरचना htp_tensor * t) {
    पटिनिवत्त t && (t->flags & HTP_TENSOR_TILED) != 0;
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_row_size(पूर्णाङ्क type, uint32_t ne00) {
    विकल्प (type) {
        पक्ष HTP_TYPE_F32:    पटिनिवत्त ne00 * 4;
        पक्ष HTP_TYPE_F16:
        पक्ष HTP_TYPE_BF16:   पटिनिवत्त ne00 * 2;
        पक्ष HTP_TYPE_Q8_0:   पटिनिवत्त (ne00 / 32) * 34;
        पक्ष HTP_TYPE_Q4_0:
        पक्ष HTP_TYPE_IQ4_NL: पटिनिवत्त (ne00 / 32) * 18;
        पक्ष HTP_TYPE_Q4_1:   पटिनिवत्त (ne00 / 32) * 20;
        पक्ष HTP_TYPE_MXFP4:  पटिनिवत्त (ne00 / 32) * 17;
        पक्ष HTP_TYPE_Q2_K:   पटिनिवत्त (ne00 / 256) * 84;
        पक्ष HTP_TYPE_Q2_0:   पटिनिवत्त (ne00 / 64) * 18;
        पक्ष HTP_TYPE_Q3_K:   पटिनिवत्त (ne00 / 256) * 110;
        पक्ष HTP_TYPE_Q4_K:   पटिनिवत्त (ne00 / 256) * 144;
        पक्ष HTP_TYPE_Q5_K:   पटिनिवत्त (ne00 / 256) * 176;
        पक्ष HTP_TYPE_Q6_K:   पटिनिवत्त (ne00 / 256) * 210;
        पक्ष HTP_TYPE_Q1_0:   पटिनिवत्त (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        पक्ष HTP_TYPE_I32:    पटिनिवत्त ne00 * 4;
        पक्ष HTP_TYPE_I64:    पटिनिवत्त ne00 * 8;
        स्वभाव:              पटिनिवत्त 0;
    }
}

#endif // HTP_TENSOR_H
