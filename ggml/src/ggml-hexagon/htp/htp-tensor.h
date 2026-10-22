#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "hex-bitmap.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE   128

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

ସଂରଚନା htp_tensor_mdev_range {
    uint32_t start;
    uint32_t ଗଣନା_2;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_tensor_is_extended(ଅଚଳ ସଂରଚନା htp_tensor * t) {
    ଫେରନ୍ତୁ t && (t->data >> 32) != 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_tensor_is_contiguous(ଅଚଳ ସଂରଚନା htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    ଯଦି (t->ne[0] != 1 && t->nb[0] != next_nb) {
        ଫେରନ୍ତୁ false;
    }
    next_nb *= t->ne[0];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < HTP_OP_MAX_DIMS; i++) {
        ଯଦି (t->ne[i] != 1 && t->nb[i] != next_nb) {
            ଫେରନ୍ତୁ false;
        }
        next_nb *= t->ne[i];
    }
    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_tensor_is_permuted(ଅଚଳ ସଂରଚନା htp_tensor * t) {
    ଫେରନ୍ତୁ t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_tensor_mdev_data_aligned(ଅଚଳ ସଂରଚନା htp_tensor * t) {
    ଫେରନ୍ତୁ ((uintptr_t) t->data & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_tensor_can_row_partition(ଅଚଳ ସଂରଚନା htp_tensor * t, uint32_t elem_size) {
    ଯଦି (!htp_tensor_mdev_data_aligned(t)) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (htp_tensor_is_permuted(t)) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) != 0) ଫେରନ୍ତୁ false;
    ଯଦି (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) != 0) ଫେରନ୍ତୁ false;
    ଯଦି (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) != 0) ଫେରନ୍ତୁ false;
    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_tensor_mdev_rows_per_chunk(ଅଚଳ ସଂରଚନା htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    ଯଦି (!htp_tensor_mdev_data_aligned(t)) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (t->ne[0] != 1 && t->nb[0] != elem_size) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (htp_tensor_is_permuted(t)) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        ଫେରନ୍ତୁ true;
    }
    ଯଦି (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE)) : 1;
        ଫେରନ୍ତୁ true;
    }
    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ସଂରଚନା htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_ଗଣନା, ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * mdev_ଗଣନା_div) {
    ସଂରଚନା htp_tensor_mdev_range range = { 0, total_units };

    ଯଦି (mdev_ଗଣନା <= 1) {
        ଫେରନ୍ତୁ range;
    }

    ଯଦି (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.ଗଣନା_2 = (mdev_idx == 0) ? total_units : 0;
        ଫେରନ୍ତୁ range;
    }

    ଅଚଳ uint32_t total_chunks = total_units / units_per_chunk;
    ଯଦି (total_chunks < mdev_ଗଣନା) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.ଗଣନା_2 = (mdev_idx == 0) ? total_units : 0;
        ଫେରନ୍ତୁ range;
    }

    ଅଚଳ uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_ଗଣନା - 1, mdev_ଗଣନା_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    ଯଦି (mdev_idx == mdev_ଗଣନା - 1) {
        range.ଗଣନା_2 = total_units - range.start;
    } ଅନ୍ୟଥା {
        range.ଗଣନା_2 = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    ଫେରନ୍ତୁ range;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t htp_tensor_get_tile_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା type) {
    ବାଛ (type) {
        ସ୍ଥିତି HTP_TYPE_Q4_0:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q4_0;
        ସ୍ଥିତି HTP_TYPE_IQ4_NL: ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_IQ4_NL;
        ସ୍ଥିତି HTP_TYPE_Q4_1:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q4_1;
        ସ୍ଥିତି HTP_TYPE_Q4_K:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q4_K;
        ସ୍ଥିତି HTP_TYPE_Q8_0:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q8_0;
        ସ୍ଥିତି HTP_TYPE_Q5_K:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q5_K;
        ସ୍ଥିତି HTP_TYPE_Q6_K:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q6_K;
        ସ୍ଥିତି HTP_TYPE_Q3_K:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q3_K;
        ସ୍ଥିତି HTP_TYPE_Q2_K:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q2_K;
        ସ୍ଥିତି HTP_TYPE_Q1_0:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q1_0;
        ସ୍ଥିତି HTP_TYPE_Q2_0:   ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_Q2_0;
        ସ୍ଥିତି HTP_TYPE_MXFP4:  ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_MXFP4;
        ସ୍ଥିତି HTP_TYPE_F16:    ଫେରନ୍ତୁ HTP_TENSOR_TILE_SIZE_F16;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:              ଫେରନ୍ତୁ 0;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t htp_tensor_get_tile_k(ପୂର୍ଣ୍ଣସଂଖ୍ୟା type) {
    ବାଛ (type) {
        ସ୍ଥିତି HTP_TYPE_Q1_0: ଫେରନ୍ତୁ HTP_TENSOR_TILE_K_Q1_0;
        ସ୍ଥିତି HTP_TYPE_Q2_0: ଫେରନ୍ତୁ HTP_TENSOR_TILE_K_Q2_0;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:            ଫେରନ୍ତୁ 32;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t htp_tensor_get_tile_padded_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା type) {
    ଫେରନ୍ତୁ (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_tensor_is_tiled(ଅଚଳ ସଂରଚନା htp_tensor * t) {
    ଫେରନ୍ତୁ t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t htp_tensor_get_row_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା type, uint32_t ne00) {
    ବାଛ (type) {
        ସ୍ଥିତି HTP_TYPE_F32:    ଫେରନ୍ତୁ ne00 * 4;
        ସ୍ଥିତି HTP_TYPE_F16:
        ସ୍ଥିତି HTP_TYPE_BF16:   ଫେରନ୍ତୁ ne00 * 2;
        ସ୍ଥିତି HTP_TYPE_Q8_0:   ଫେରନ୍ତୁ (ne00 / 32) * 34;
        ସ୍ଥିତି HTP_TYPE_Q4_0:
        ସ୍ଥିତି HTP_TYPE_IQ4_NL: ଫେରନ୍ତୁ (ne00 / 32) * 18;
        ସ୍ଥିତି HTP_TYPE_Q4_1:   ଫେରନ୍ତୁ (ne00 / 32) * 20;
        ସ୍ଥିତି HTP_TYPE_MXFP4:  ଫେରନ୍ତୁ (ne00 / 32) * 17;
        ସ୍ଥିତି HTP_TYPE_Q2_K:   ଫେରନ୍ତୁ (ne00 / 256) * 84;
        ସ୍ଥିତି HTP_TYPE_Q2_0:   ଫେରନ୍ତୁ (ne00 / 64) * 18;
        ସ୍ଥିତି HTP_TYPE_Q3_K:   ଫେରନ୍ତୁ (ne00 / 256) * 110;
        ସ୍ଥିତି HTP_TYPE_Q4_K:   ଫେରନ୍ତୁ (ne00 / 256) * 144;
        ସ୍ଥିତି HTP_TYPE_Q5_K:   ଫେରନ୍ତୁ (ne00 / 256) * 176;
        ସ୍ଥିତି HTP_TYPE_Q6_K:   ଫେରନ୍ତୁ (ne00 / 256) * 210;
        ସ୍ଥିତି HTP_TYPE_Q1_0:   ଫେରନ୍ତୁ (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        ସ୍ଥିତି HTP_TYPE_I32:    ଫେରନ୍ତୁ ne00 * 4;
        ସ୍ଥିତି HTP_TYPE_I64:    ଫେରନ୍ତୁ ne00 * 8;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:              ଫେରନ୍ତୁ 0;
    }
}

#endif // HTP_TENSOR_H
