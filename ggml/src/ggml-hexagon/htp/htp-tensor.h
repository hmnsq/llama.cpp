#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن "htp-ops.h"
#ضمن "hex-bitmap.h"
#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_قۇر_SIZE   128

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

بنية htp_tensor_mdev_range {
    uint32_t start;
    uint32_t سانى;
};

ساكن مضمن bool htp_tensor_is_extended(ثابت بنية htp_tensor * t) {
    أرجع t && (t->data >> 32) != 0;
}

ساكن مضمن bool htp_tensor_is_contiguous(ثابت بنية htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    إذا (t->ne[0] != 1 && t->nb[0] != next_nb) {
        أرجع false;
    }
    next_nb *= t->ne[0];
    لكل (صحيح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        إذا (t->ne[i] != 1 && t->nb[i] != next_nb) {
            أرجع false;
        }
        next_nb *= t->ne[i];
    }
    أرجع true;
}

ساكن مضمن bool htp_tensor_is_permuted(ثابت بنية htp_tensor * t) {
    أرجع t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساكن مضمن bool htp_tensor_mdev_data_aligned(ثابت بنية htp_tensor * t) {
    أرجع ((uintptr_t) t->data & (HTP_TENSOR_MDEV_قۇر_SIZE - 1)) == 0;
}

ساكن مضمن bool htp_tensor_can_row_partition(ثابت بنية htp_tensor * t, uint32_t elem_size) {
    إذا (!htp_tensor_mdev_data_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_size) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_قۇر_SIZE - 1)) != 0) أرجع false;
    إذا (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_قۇر_SIZE - 1)) != 0) أرجع false;
    إذا (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_قۇر_SIZE - 1)) != 0) أرجع false;
    أرجع true;
}

ساكن مضمن bool htp_tensor_mdev_rows_per_chunk(ثابت بنية htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    إذا (!htp_tensor_mdev_data_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_size) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_قۇر_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_قۇر_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_قۇر_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        أرجع true;
    }
    إذا (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_قۇر_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_قۇر_SIZE)) : 1;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن بنية htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_سانى, ثابت بنية fastdiv_قىممەتلەر * mdev_سانى_div) {
    بنية htp_tensor_mdev_range range = { 0, total_units };

    إذا (mdev_سانى <= 1) {
        أرجع range;
    }

    إذا (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.سانى = (mdev_idx == 0) ? total_units : 0;
        أرجع range;
    }

    ثابت uint32_t total_chunks = total_units / units_per_chunk;
    إذا (total_chunks < mdev_سانى) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.سانى = (mdev_idx == 0) ? total_units : 0;
        أرجع range;
    }

    ثابت uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_سانى - 1, mdev_سانى_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    إذا (mdev_idx == mdev_سانى - 1) {
        range.سانى = total_units - range.start;
    } وإلا {
        range.سانى = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    أرجع range;
}

ساكن مضمن uint32_t htp_tensor_get_tile_size(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:   أرجع HTP_TENSOR_TILE_SIZE_Q4_0;
        حالة HTP_TYPE_IQ4_NL: أرجع HTP_TENSOR_TILE_SIZE_IQ4_NL;
        حالة HTP_TYPE_Q4_1:   أرجع HTP_TENSOR_TILE_SIZE_Q4_1;
        حالة HTP_TYPE_Q4_K:   أرجع HTP_TENSOR_TILE_SIZE_Q4_K;
        حالة HTP_TYPE_Q8_0:   أرجع HTP_TENSOR_TILE_SIZE_Q8_0;
        حالة HTP_TYPE_Q5_K:   أرجع HTP_TENSOR_TILE_SIZE_Q5_K;
        حالة HTP_TYPE_Q6_K:   أرجع HTP_TENSOR_TILE_SIZE_Q6_K;
        حالة HTP_TYPE_Q3_K:   أرجع HTP_TENSOR_TILE_SIZE_Q3_K;
        حالة HTP_TYPE_Q2_K:   أرجع HTP_TENSOR_TILE_SIZE_Q2_K;
        حالة HTP_TYPE_Q1_0:   أرجع HTP_TENSOR_TILE_SIZE_Q1_0;
        حالة HTP_TYPE_Q2_0:   أرجع HTP_TENSOR_TILE_SIZE_Q2_0;
        حالة HTP_TYPE_MXFP4:  أرجع HTP_TENSOR_TILE_SIZE_MXFP4;
        حالة HTP_TYPE_F16:    أرجع HTP_TENSOR_TILE_SIZE_F16;
        افتراضي:              أرجع 0;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_k(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q1_0: أرجع HTP_TENSOR_TILE_K_Q1_0;
        حالة HTP_TYPE_Q2_0: أرجع HTP_TENSOR_TILE_K_Q2_0;
        افتراضي:            أرجع 32;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_padded_size(صحيح type) {
    أرجع (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

ساكن مضمن bool htp_tensor_is_tiled(ثابت بنية htp_tensor * t) {
    أرجع t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ساكن مضمن uint32_t htp_tensor_get_row_size(صحيح type, uint32_t ne00) {
    بدل (type) {
        حالة HTP_TYPE_F32:    أرجع ne00 * 4;
        حالة HTP_TYPE_F16:
        حالة HTP_TYPE_BF16:   أرجع ne00 * 2;
        حالة HTP_TYPE_Q8_0:   أرجع (ne00 / 32) * 34;
        حالة HTP_TYPE_Q4_0:
        حالة HTP_TYPE_IQ4_NL: أرجع (ne00 / 32) * 18;
        حالة HTP_TYPE_Q4_1:   أرجع (ne00 / 32) * 20;
        حالة HTP_TYPE_MXFP4:  أرجع (ne00 / 32) * 17;
        حالة HTP_TYPE_Q2_K:   أرجع (ne00 / 256) * 84;
        حالة HTP_TYPE_Q2_0:   أرجع (ne00 / 64) * 18;
        حالة HTP_TYPE_Q3_K:   أرجع (ne00 / 256) * 110;
        حالة HTP_TYPE_Q4_K:   أرجع (ne00 / 256) * 144;
        حالة HTP_TYPE_Q5_K:   أرجع (ne00 / 256) * 176;
        حالة HTP_TYPE_Q6_K:   أرجع (ne00 / 256) * 210;
        حالة HTP_TYPE_Q1_0:   أرجع (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        حالة HTP_TYPE_I32:    أرجع ne00 * 4;
        حالة HTP_TYPE_I64:    أرجع ne00 * 8;
        افتراضي:              أرجع 0;
    }
}

#endif // HTP_TENSOR_H
