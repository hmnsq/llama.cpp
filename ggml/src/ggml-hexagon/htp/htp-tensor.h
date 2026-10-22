#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "hex-bitmap.h"
#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_લીટી_SIZE   128

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

સંરચના htp_tensor_mdev_range {
    uint32_t start;
    uint32_t ગણતરી;
};

સ્થિર અંતઃરેખ bool htp_tensor_is_extended(અચલ સંરચના htp_tensor * t) {
    પાછા t && (t->data >> 32) != 0;
}

સ્થિર અંતઃરેખ bool htp_tensor_is_contiguous(અચલ સંરચના htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    જો (t->ne[0] != 1 && t->nb[0] != next_nb) {
        પાછા false;
    }
    next_nb *= t->ne[0];
    માટે (પૂર્ણાંક i = 1; i < HTP_OP_MAX_DIMS; i++) {
        જો (t->ne[i] != 1 && t->nb[i] != next_nb) {
            પાછા false;
        }
        next_nb *= t->ne[i];
    }
    પાછા true;
}

સ્થિર અંતઃરેખ bool htp_tensor_is_permuted(અચલ સંરચના htp_tensor * t) {
    પાછા t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

સ્થિર અંતઃરેખ bool htp_tensor_mdev_data_aligned(અચલ સંરચના htp_tensor * t) {
    પાછા ((uintptr_t) t->data & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) == 0;
}

સ્થિર અંતઃરેખ bool htp_tensor_can_row_partition(અચલ સંરચના htp_tensor * t, uint32_t elem_size) {
    જો (!htp_tensor_mdev_data_aligned(t)) {
        પાછા false;
    }
    જો (t->ne[0] != 1 && t->nb[0] != elem_size) {
        પાછા false;
    }
    જો (htp_tensor_is_permuted(t)) {
        પાછા false;
    }
    જો (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) != 0) પાછા false;
    જો (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) != 0) પાછા false;
    જો (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) != 0) પાછા false;
    પાછા true;
}

સ્થિર અંતઃરેખ bool htp_tensor_mdev_rows_per_chunk(અચલ સંરચના htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    જો (!htp_tensor_mdev_data_aligned(t)) {
        પાછા false;
    }
    જો (t->ne[0] != 1 && t->nb[0] != elem_size) {
        પાછા false;
    }
    જો (htp_tensor_is_permuted(t)) {
        પાછા false;
    }
    જો (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        પાછા true;
    }
    જો (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_લીટી_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_લીટી_SIZE)) : 1;
        પાછા true;
    }
    પાછા false;
}

સ્થિર અંતઃરેખ સંરચના htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_ગણતરી, અચલ સંરચના fastdiv_કિંમતો * mdev_ગણતરી_div) {
    સંરચના htp_tensor_mdev_range range = { 0, total_units };

    જો (mdev_ગણતરી <= 1) {
        પાછા range;
    }

    જો (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.ગણતરી = (mdev_idx == 0) ? total_units : 0;
        પાછા range;
    }

    અચલ uint32_t total_chunks = total_units / units_per_chunk;
    જો (total_chunks < mdev_ગણતરી) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.ગણતરી = (mdev_idx == 0) ? total_units : 0;
        પાછા range;
    }

    અચલ uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_ગણતરી - 1, mdev_ગણતરી_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    જો (mdev_idx == mdev_ગણતરી - 1) {
        range.ગણતરી = total_units - range.start;
    } નહીંતર {
        range.ગણતરી = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    પાછા range;
}

સ્થિર અંતઃરેખ uint32_t htp_tensor_get_tile_size(પૂર્ણાંક type) {
    પસંદગી (type) {
        સ્થિતિ HTP_TYPE_Q4_0:   પાછા HTP_TENSOR_TILE_SIZE_Q4_0;
        સ્થિતિ HTP_TYPE_IQ4_NL: પાછા HTP_TENSOR_TILE_SIZE_IQ4_NL;
        સ્થિતિ HTP_TYPE_Q4_1:   પાછા HTP_TENSOR_TILE_SIZE_Q4_1;
        સ્થિતિ HTP_TYPE_Q4_K:   પાછા HTP_TENSOR_TILE_SIZE_Q4_K;
        સ્થિતિ HTP_TYPE_Q8_0:   પાછા HTP_TENSOR_TILE_SIZE_Q8_0;
        સ્થિતિ HTP_TYPE_Q5_K:   પાછા HTP_TENSOR_TILE_SIZE_Q5_K;
        સ્થિતિ HTP_TYPE_Q6_K:   પાછા HTP_TENSOR_TILE_SIZE_Q6_K;
        સ્થિતિ HTP_TYPE_Q3_K:   પાછા HTP_TENSOR_TILE_SIZE_Q3_K;
        સ્થિતિ HTP_TYPE_Q2_K:   પાછા HTP_TENSOR_TILE_SIZE_Q2_K;
        સ્થિતિ HTP_TYPE_Q1_0:   પાછા HTP_TENSOR_TILE_SIZE_Q1_0;
        સ્થિતિ HTP_TYPE_Q2_0:   પાછા HTP_TENSOR_TILE_SIZE_Q2_0;
        સ્થિતિ HTP_TYPE_MXFP4:  પાછા HTP_TENSOR_TILE_SIZE_MXFP4;
        સ્થિતિ HTP_TYPE_F16:    પાછા HTP_TENSOR_TILE_SIZE_F16;
        મૂળભૂત:              પાછા 0;
    }
}

સ્થિર અંતઃરેખ uint32_t htp_tensor_get_tile_k(પૂર્ણાંક type) {
    પસંદગી (type) {
        સ્થિતિ HTP_TYPE_Q1_0: પાછા HTP_TENSOR_TILE_K_Q1_0;
        સ્થિતિ HTP_TYPE_Q2_0: પાછા HTP_TENSOR_TILE_K_Q2_0;
        મૂળભૂત:            પાછા 32;
    }
}

સ્થિર અંતઃરેખ uint32_t htp_tensor_get_tile_padded_size(પૂર્ણાંક type) {
    પાછા (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

સ્થિર અંતઃરેખ bool htp_tensor_is_tiled(અચલ સંરચના htp_tensor * t) {
    પાછા t && (t->flags & HTP_TENSOR_TILED) != 0;
}

સ્થિર અંતઃરેખ uint32_t htp_tensor_get_row_size(પૂર્ણાંક type, uint32_t ne00) {
    પસંદગી (type) {
        સ્થિતિ HTP_TYPE_F32:    પાછા ne00 * 4;
        સ્થિતિ HTP_TYPE_F16:
        સ્થિતિ HTP_TYPE_BF16:   પાછા ne00 * 2;
        સ્થિતિ HTP_TYPE_Q8_0:   પાછા (ne00 / 32) * 34;
        સ્થિતિ HTP_TYPE_Q4_0:
        સ્થિતિ HTP_TYPE_IQ4_NL: પાછા (ne00 / 32) * 18;
        સ્થિતિ HTP_TYPE_Q4_1:   પાછા (ne00 / 32) * 20;
        સ્થિતિ HTP_TYPE_MXFP4:  પાછા (ne00 / 32) * 17;
        સ્થિતિ HTP_TYPE_Q2_K:   પાછા (ne00 / 256) * 84;
        સ્થિતિ HTP_TYPE_Q2_0:   પાછા (ne00 / 64) * 18;
        સ્થિતિ HTP_TYPE_Q3_K:   પાછા (ne00 / 256) * 110;
        સ્થિતિ HTP_TYPE_Q4_K:   પાછા (ne00 / 256) * 144;
        સ્થિતિ HTP_TYPE_Q5_K:   પાછા (ne00 / 256) * 176;
        સ્થિતિ HTP_TYPE_Q6_K:   પાછા (ne00 / 256) * 210;
        સ્થિતિ HTP_TYPE_Q1_0:   પાછા (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        સ્થિતિ HTP_TYPE_I32:    પાછા ne00 * 4;
        સ્થિતિ HTP_TYPE_I64:    પાછા ne00 * 8;
        મૂળભૂત:              પાછા 0;
    }
}

#endif // HTP_TENSOR_H
