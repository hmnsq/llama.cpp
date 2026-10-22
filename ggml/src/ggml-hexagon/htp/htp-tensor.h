#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "hex-bitmap.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_সারি_SIZE   128

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

সংরচনা htp_tensor_mdev_range {
    uint32_t start;
    uint32_t গুনতি;
};

স্থির অন্তঃস্থ bool htp_tensor_is_extended(ধ্রুব সংরচনা htp_tensor * t) {
    প্রত্যাগম t && (t->data >> 32) != 0;
}

স্থির অন্তঃস্থ bool htp_tensor_is_contiguous(ধ্রুব সংরচনা htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    যদি (t->ne[0] != 1 && t->nb[0] != next_nb) {
        প্রত্যাগম false;
    }
    next_nb *= t->ne[0];
    প্রতি (পূর্ণাঙ্ক i = 1; i < HTP_OP_MAX_DIMS; i++) {
        যদি (t->ne[i] != 1 && t->nb[i] != next_nb) {
            প্রত্যাগম false;
        }
        next_nb *= t->ne[i];
    }
    প্রত্যাগম true;
}

স্থির অন্তঃস্থ bool htp_tensor_is_permuted(ধ্রুব সংরচনা htp_tensor * t) {
    প্রত্যাগম t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

স্থির অন্তঃস্থ bool htp_tensor_mdev_data_aligned(ধ্রুব সংরচনা htp_tensor * t) {
    প্রত্যাগম ((uintptr_t) t->data & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) == 0;
}

স্থির অন্তঃস্থ bool htp_tensor_can_row_partition(ধ্রুব সংরচনা htp_tensor * t, uint32_t elem_size) {
    যদি (!htp_tensor_mdev_data_aligned(t)) {
        প্রত্যাগম false;
    }
    যদি (t->ne[0] != 1 && t->nb[0] != elem_size) {
        প্রত্যাগম false;
    }
    যদি (htp_tensor_is_permuted(t)) {
        প্রত্যাগম false;
    }
    যদি (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) != 0) প্রত্যাগম false;
    যদি (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) != 0) প্রত্যাগম false;
    যদি (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) != 0) প্রত্যাগম false;
    প্রত্যাগম true;
}

স্থির অন্তঃস্থ bool htp_tensor_mdev_rows_per_chunk(ধ্রুব সংরচনা htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    যদি (!htp_tensor_mdev_data_aligned(t)) {
        প্রত্যাগম false;
    }
    যদি (t->ne[0] != 1 && t->nb[0] != elem_size) {
        প্রত্যাগম false;
    }
    যদি (htp_tensor_is_permuted(t)) {
        প্রত্যাগম false;
    }
    যদি (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        প্রত্যাগম true;
    }
    যদি (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_সারি_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_সারি_SIZE)) : 1;
        প্রত্যাগম true;
    }
    প্রত্যাগম false;
}

স্থির অন্তঃস্থ সংরচনা htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_গুনতি, ধ্রুব সংরচনা fastdiv_মানসমূহ * mdev_গুনতি_div) {
    সংরচনা htp_tensor_mdev_range range = { 0, total_units };

    যদি (mdev_গুনতি <= 1) {
        প্রত্যাগম range;
    }

    যদি (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.গুনতি = (mdev_idx == 0) ? total_units : 0;
        প্রত্যাগম range;
    }

    ধ্রুব uint32_t total_chunks = total_units / units_per_chunk;
    যদি (total_chunks < mdev_গুনতি) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.গুনতি = (mdev_idx == 0) ? total_units : 0;
        প্রত্যাগম range;
    }

    ধ্রুব uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_গুনতি - 1, mdev_গুনতি_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    যদি (mdev_idx == mdev_গুনতি - 1) {
        range.গুনতি = total_units - range.start;
    } অন্যথা {
        range.গুনতি = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    প্রত্যাগম range;
}

স্থির অন্তঃস্থ uint32_t htp_tensor_get_tile_size(পূর্ণাঙ্ক type) {
    বিকল্প (type) {
        পক্ষ HTP_TYPE_Q4_0:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q4_0;
        পক্ষ HTP_TYPE_IQ4_NL: প্রত্যাগম HTP_TENSOR_TILE_SIZE_IQ4_NL;
        পক্ষ HTP_TYPE_Q4_1:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q4_1;
        পক্ষ HTP_TYPE_Q4_K:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q4_K;
        পক্ষ HTP_TYPE_Q8_0:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q8_0;
        পক্ষ HTP_TYPE_Q5_K:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q5_K;
        পক্ষ HTP_TYPE_Q6_K:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q6_K;
        পক্ষ HTP_TYPE_Q3_K:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q3_K;
        পক্ষ HTP_TYPE_Q2_K:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q2_K;
        পক্ষ HTP_TYPE_Q1_0:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q1_0;
        পক্ষ HTP_TYPE_Q2_0:   প্রত্যাগম HTP_TENSOR_TILE_SIZE_Q2_0;
        পক্ষ HTP_TYPE_MXFP4:  প্রত্যাগম HTP_TENSOR_TILE_SIZE_MXFP4;
        পক্ষ HTP_TYPE_F16:    প্রত্যাগম HTP_TENSOR_TILE_SIZE_F16;
        স্বভাব:              প্রত্যাগম 0;
    }
}

স্থির অন্তঃস্থ uint32_t htp_tensor_get_tile_k(পূর্ণাঙ্ক type) {
    বিকল্প (type) {
        পক্ষ HTP_TYPE_Q1_0: প্রত্যাগম HTP_TENSOR_TILE_K_Q1_0;
        পক্ষ HTP_TYPE_Q2_0: প্রত্যাগম HTP_TENSOR_TILE_K_Q2_0;
        স্বভাব:            প্রত্যাগম 32;
    }
}

স্থির অন্তঃস্থ uint32_t htp_tensor_get_tile_padded_size(পূর্ণাঙ্ক type) {
    প্রত্যাগম (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

স্থির অন্তঃস্থ bool htp_tensor_is_tiled(ধ্রুব সংরচনা htp_tensor * t) {
    প্রত্যাগম t && (t->flags & HTP_TENSOR_TILED) != 0;
}

স্থির অন্তঃস্থ uint32_t htp_tensor_get_row_size(পূর্ণাঙ্ক type, uint32_t ne00) {
    বিকল্প (type) {
        পক্ষ HTP_TYPE_F32:    প্রত্যাগম ne00 * 4;
        পক্ষ HTP_TYPE_F16:
        পক্ষ HTP_TYPE_BF16:   প্রত্যাগম ne00 * 2;
        পক্ষ HTP_TYPE_Q8_0:   প্রত্যাগম (ne00 / 32) * 34;
        পক্ষ HTP_TYPE_Q4_0:
        পক্ষ HTP_TYPE_IQ4_NL: প্রত্যাগম (ne00 / 32) * 18;
        পক্ষ HTP_TYPE_Q4_1:   প্রত্যাগম (ne00 / 32) * 20;
        পক্ষ HTP_TYPE_MXFP4:  প্রত্যাগম (ne00 / 32) * 17;
        পক্ষ HTP_TYPE_Q2_K:   প্রত্যাগম (ne00 / 256) * 84;
        পক্ষ HTP_TYPE_Q2_0:   প্রত্যাগম (ne00 / 64) * 18;
        পক্ষ HTP_TYPE_Q3_K:   প্রত্যাগম (ne00 / 256) * 110;
        পক্ষ HTP_TYPE_Q4_K:   প্রত্যাগম (ne00 / 256) * 144;
        পক্ষ HTP_TYPE_Q5_K:   প্রত্যাগম (ne00 / 256) * 176;
        পক্ষ HTP_TYPE_Q6_K:   প্রত্যাগম (ne00 / 256) * 210;
        পক্ষ HTP_TYPE_Q1_0:   প্রত্যাগম (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        পক্ষ HTP_TYPE_I32:    প্রত্যাগম ne00 * 4;
        পক্ষ HTP_TYPE_I64:    প্রত্যাগম ne00 * 8;
        স্বভাব:              প্রত্যাগম 0;
    }
}

#endif // HTP_TENSOR_H
