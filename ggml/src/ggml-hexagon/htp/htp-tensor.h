#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "hex-bitmap.h"
#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_വരി_SIZE   128

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

ഘടന htp_tensor_mdev_range {
    uint32_t start;
    uint32_t എണ്ണം;
};

സ്ഥിതം വരിയിൽ bool htp_tensor_is_extended(സ്ഥിരം ഘടന htp_tensor * t) {
    തിരികെ t && (t->data >> 32) != 0;
}

സ്ഥിതം വരിയിൽ bool htp_tensor_is_contiguous(സ്ഥിരം ഘടന htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    എങ്കിൽ (t->ne[0] != 1 && t->nb[0] != next_nb) {
        തിരികെ false;
    }
    next_nb *= t->ne[0];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < HTP_OP_MAX_DIMS; i++) {
        എങ്കിൽ (t->ne[i] != 1 && t->nb[i] != next_nb) {
            തിരികെ false;
        }
        next_nb *= t->ne[i];
    }
    തിരികെ true;
}

സ്ഥിതം വരിയിൽ bool htp_tensor_is_permuted(സ്ഥിരം ഘടന htp_tensor * t) {
    തിരികെ t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

സ്ഥിതം വരിയിൽ bool htp_tensor_mdev_data_aligned(സ്ഥിരം ഘടന htp_tensor * t) {
    തിരികെ ((uintptr_t) t->data & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) == 0;
}

സ്ഥിതം വരിയിൽ bool htp_tensor_can_row_partition(സ്ഥിരം ഘടന htp_tensor * t, uint32_t elem_size) {
    എങ്കിൽ (!htp_tensor_mdev_data_aligned(t)) {
        തിരികെ false;
    }
    എങ്കിൽ (t->ne[0] != 1 && t->nb[0] != elem_size) {
        തിരികെ false;
    }
    എങ്കിൽ (htp_tensor_is_permuted(t)) {
        തിരികെ false;
    }
    എങ്കിൽ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) != 0) തിരികെ false;
    എങ്കിൽ (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) != 0) തിരികെ false;
    എങ്കിൽ (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) != 0) തിരികെ false;
    തിരികെ true;
}

സ്ഥിതം വരിയിൽ bool htp_tensor_mdev_rows_per_chunk(സ്ഥിരം ഘടന htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    എങ്കിൽ (!htp_tensor_mdev_data_aligned(t)) {
        തിരികെ false;
    }
    എങ്കിൽ (t->ne[0] != 1 && t->nb[0] != elem_size) {
        തിരികെ false;
    }
    എങ്കിൽ (htp_tensor_is_permuted(t)) {
        തിരികെ false;
    }
    എങ്കിൽ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        തിരികെ true;
    }
    എങ്കിൽ (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_വരി_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_വരി_SIZE)) : 1;
        തിരികെ true;
    }
    തിരികെ false;
}

സ്ഥിതം വരിയിൽ ഘടന htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_എണ്ണം, സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * mdev_എണ്ണം_div) {
    ഘടന htp_tensor_mdev_range range = { 0, total_units };

    എങ്കിൽ (mdev_എണ്ണം <= 1) {
        തിരികെ range;
    }

    എങ്കിൽ (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.എണ്ണം = (mdev_idx == 0) ? total_units : 0;
        തിരികെ range;
    }

    സ്ഥിരം uint32_t total_chunks = total_units / units_per_chunk;
    എങ്കിൽ (total_chunks < mdev_എണ്ണം) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.എണ്ണം = (mdev_idx == 0) ? total_units : 0;
        തിരികെ range;
    }

    സ്ഥിരം uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_എണ്ണം - 1, mdev_എണ്ണം_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    എങ്കിൽ (mdev_idx == mdev_എണ്ണം - 1) {
        range.എണ്ണം = total_units - range.start;
    } അല്ലെങ്കിൽ {
        range.എണ്ണം = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    തിരികെ range;
}

സ്ഥിതം വരിയിൽ uint32_t htp_tensor_get_tile_size(പൂർണ്ണസംഖ്യ type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ HTP_TYPE_Q4_0:   തിരികെ HTP_TENSOR_TILE_SIZE_Q4_0;
        അവസ്ഥ HTP_TYPE_IQ4_NL: തിരികെ HTP_TENSOR_TILE_SIZE_IQ4_NL;
        അവസ്ഥ HTP_TYPE_Q4_1:   തിരികെ HTP_TENSOR_TILE_SIZE_Q4_1;
        അവസ്ഥ HTP_TYPE_Q4_K:   തിരികെ HTP_TENSOR_TILE_SIZE_Q4_K;
        അവസ്ഥ HTP_TYPE_Q8_0:   തിരികെ HTP_TENSOR_TILE_SIZE_Q8_0;
        അവസ്ഥ HTP_TYPE_Q5_K:   തിരികെ HTP_TENSOR_TILE_SIZE_Q5_K;
        അവസ്ഥ HTP_TYPE_Q6_K:   തിരികെ HTP_TENSOR_TILE_SIZE_Q6_K;
        അവസ്ഥ HTP_TYPE_Q3_K:   തിരികെ HTP_TENSOR_TILE_SIZE_Q3_K;
        അവസ്ഥ HTP_TYPE_Q2_K:   തിരികെ HTP_TENSOR_TILE_SIZE_Q2_K;
        അവസ്ഥ HTP_TYPE_Q1_0:   തിരികെ HTP_TENSOR_TILE_SIZE_Q1_0;
        അവസ്ഥ HTP_TYPE_Q2_0:   തിരികെ HTP_TENSOR_TILE_SIZE_Q2_0;
        അവസ്ഥ HTP_TYPE_MXFP4:  തിരികെ HTP_TENSOR_TILE_SIZE_MXFP4;
        അവസ്ഥ HTP_TYPE_F16:    തിരികെ HTP_TENSOR_TILE_SIZE_F16;
        സ്ഥിരസ്ഥിതി:              തിരികെ 0;
    }
}

സ്ഥിതം വരിയിൽ uint32_t htp_tensor_get_tile_k(പൂർണ്ണസംഖ്യ type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ HTP_TYPE_Q1_0: തിരികെ HTP_TENSOR_TILE_K_Q1_0;
        അവസ്ഥ HTP_TYPE_Q2_0: തിരികെ HTP_TENSOR_TILE_K_Q2_0;
        സ്ഥിരസ്ഥിതി:            തിരികെ 32;
    }
}

സ്ഥിതം വരിയിൽ uint32_t htp_tensor_get_tile_padded_size(പൂർണ്ണസംഖ്യ type) {
    തിരികെ (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

സ്ഥിതം വരിയിൽ bool htp_tensor_is_tiled(സ്ഥിരം ഘടന htp_tensor * t) {
    തിരികെ t && (t->flags & HTP_TENSOR_TILED) != 0;
}

സ്ഥിതം വരിയിൽ uint32_t htp_tensor_get_row_size(പൂർണ്ണസംഖ്യ type, uint32_t ne00) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ HTP_TYPE_F32:    തിരികെ ne00 * 4;
        അവസ്ഥ HTP_TYPE_F16:
        അവസ്ഥ HTP_TYPE_BF16:   തിരികെ ne00 * 2;
        അവസ്ഥ HTP_TYPE_Q8_0:   തിരികെ (ne00 / 32) * 34;
        അവസ്ഥ HTP_TYPE_Q4_0:
        അവസ്ഥ HTP_TYPE_IQ4_NL: തിരികെ (ne00 / 32) * 18;
        അവസ്ഥ HTP_TYPE_Q4_1:   തിരികെ (ne00 / 32) * 20;
        അവസ്ഥ HTP_TYPE_MXFP4:  തിരികെ (ne00 / 32) * 17;
        അവസ്ഥ HTP_TYPE_Q2_K:   തിരികെ (ne00 / 256) * 84;
        അവസ്ഥ HTP_TYPE_Q2_0:   തിരികെ (ne00 / 64) * 18;
        അവസ്ഥ HTP_TYPE_Q3_K:   തിരികെ (ne00 / 256) * 110;
        അവസ്ഥ HTP_TYPE_Q4_K:   തിരികെ (ne00 / 256) * 144;
        അവസ്ഥ HTP_TYPE_Q5_K:   തിരികെ (ne00 / 256) * 176;
        അവസ്ഥ HTP_TYPE_Q6_K:   തിരികെ (ne00 / 256) * 210;
        അവസ്ഥ HTP_TYPE_Q1_0:   തിരികെ (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        അവസ്ഥ HTP_TYPE_I32:    തിരികെ ne00 * 4;
        അവസ്ഥ HTP_TYPE_I64:    തിരികെ ne00 * 8;
        സ്ഥിരസ്ഥിതി:              തിരികെ 0;
    }
}

#endif // HTP_TENSOR_H
