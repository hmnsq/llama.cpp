#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "hex-bitmap.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_ओळ_SIZE   128

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
    uint32_t मोजणी;
};

स्थिर ओळीत bool htp_tensor_is_extended(स्थिरांक संरचना htp_tensor * t) {
    परत t && (t->data >> 32) != 0;
}

स्थिर ओळीत bool htp_tensor_is_contiguous(स्थिरांक संरचना htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    जर (t->ne[0] != 1 && t->nb[0] != next_nb) {
        परत false;
    }
    next_nb *= t->ne[0];
    साठी (पूर्णांक i = 1; i < HTP_OP_MAX_DIMS; i++) {
        जर (t->ne[i] != 1 && t->nb[i] != next_nb) {
            परत false;
        }
        next_nb *= t->ne[i];
    }
    परत true;
}

स्थिर ओळीत bool htp_tensor_is_permuted(स्थिरांक संरचना htp_tensor * t) {
    परत t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

स्थिर ओळीत bool htp_tensor_mdev_data_aligned(स्थिरांक संरचना htp_tensor * t) {
    परत ((uintptr_t) t->data & (HTP_TENSOR_MDEV_ओळ_SIZE - 1)) == 0;
}

स्थिर ओळीत bool htp_tensor_can_row_partition(स्थिरांक संरचना htp_tensor * t, uint32_t elem_size) {
    जर (!htp_tensor_mdev_data_aligned(t)) {
        परत false;
    }
    जर (t->ne[0] != 1 && t->nb[0] != elem_size) {
        परत false;
    }
    जर (htp_tensor_is_permuted(t)) {
        परत false;
    }
    जर (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ओळ_SIZE - 1)) != 0) परत false;
    जर (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_ओळ_SIZE - 1)) != 0) परत false;
    जर (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_ओळ_SIZE - 1)) != 0) परत false;
    परत true;
}

स्थिर ओळीत bool htp_tensor_mdev_rows_per_chunk(स्थिरांक संरचना htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    जर (!htp_tensor_mdev_data_aligned(t)) {
        परत false;
    }
    जर (t->ne[0] != 1 && t->nb[0] != elem_size) {
        परत false;
    }
    जर (htp_tensor_is_permuted(t)) {
        परत false;
    }
    जर (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ओळ_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_ओळ_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_ओळ_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        परत true;
    }
    जर (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_ओळ_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_ओळ_SIZE)) : 1;
        परत true;
    }
    परत false;
}

स्थिर ओळीत संरचना htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_मोजणी, स्थिरांक संरचना fastdiv_मूल्ये * mdev_मोजणी_div) {
    संरचना htp_tensor_mdev_range range = { 0, total_units };

    जर (mdev_मोजणी <= 1) {
        परत range;
    }

    जर (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.मोजणी = (mdev_idx == 0) ? total_units : 0;
        परत range;
    }

    स्थिरांक uint32_t total_chunks = total_units / units_per_chunk;
    जर (total_chunks < mdev_मोजणी) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.मोजणी = (mdev_idx == 0) ? total_units : 0;
        परत range;
    }

    स्थिरांक uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_मोजणी - 1, mdev_मोजणी_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    जर (mdev_idx == mdev_मोजणी - 1) {
        range.मोजणी = total_units - range.start;
    } अन्यथा {
        range.मोजणी = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    परत range;
}

स्थिर ओळीत uint32_t htp_tensor_get_tile_size(पूर्णांक type) {
    निवड (type) {
        स्थिती HTP_TYPE_Q4_0:   परत HTP_TENSOR_TILE_SIZE_Q4_0;
        स्थिती HTP_TYPE_IQ4_NL: परत HTP_TENSOR_TILE_SIZE_IQ4_NL;
        स्थिती HTP_TYPE_Q4_1:   परत HTP_TENSOR_TILE_SIZE_Q4_1;
        स्थिती HTP_TYPE_Q4_K:   परत HTP_TENSOR_TILE_SIZE_Q4_K;
        स्थिती HTP_TYPE_Q8_0:   परत HTP_TENSOR_TILE_SIZE_Q8_0;
        स्थिती HTP_TYPE_Q5_K:   परत HTP_TENSOR_TILE_SIZE_Q5_K;
        स्थिती HTP_TYPE_Q6_K:   परत HTP_TENSOR_TILE_SIZE_Q6_K;
        स्थिती HTP_TYPE_Q3_K:   परत HTP_TENSOR_TILE_SIZE_Q3_K;
        स्थिती HTP_TYPE_Q2_K:   परत HTP_TENSOR_TILE_SIZE_Q2_K;
        स्थिती HTP_TYPE_Q1_0:   परत HTP_TENSOR_TILE_SIZE_Q1_0;
        स्थिती HTP_TYPE_Q2_0:   परत HTP_TENSOR_TILE_SIZE_Q2_0;
        स्थिती HTP_TYPE_MXFP4:  परत HTP_TENSOR_TILE_SIZE_MXFP4;
        स्थिती HTP_TYPE_F16:    परत HTP_TENSOR_TILE_SIZE_F16;
        पूर्वनिर्धारित:              परत 0;
    }
}

स्थिर ओळीत uint32_t htp_tensor_get_tile_k(पूर्णांक type) {
    निवड (type) {
        स्थिती HTP_TYPE_Q1_0: परत HTP_TENSOR_TILE_K_Q1_0;
        स्थिती HTP_TYPE_Q2_0: परत HTP_TENSOR_TILE_K_Q2_0;
        पूर्वनिर्धारित:            परत 32;
    }
}

स्थिर ओळीत uint32_t htp_tensor_get_tile_padded_size(पूर्णांक type) {
    परत (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

स्थिर ओळीत bool htp_tensor_is_tiled(स्थिरांक संरचना htp_tensor * t) {
    परत t && (t->flags & HTP_TENSOR_TILED) != 0;
}

स्थिर ओळीत uint32_t htp_tensor_get_row_size(पूर्णांक type, uint32_t ne00) {
    निवड (type) {
        स्थिती HTP_TYPE_F32:    परत ne00 * 4;
        स्थिती HTP_TYPE_F16:
        स्थिती HTP_TYPE_BF16:   परत ne00 * 2;
        स्थिती HTP_TYPE_Q8_0:   परत (ne00 / 32) * 34;
        स्थिती HTP_TYPE_Q4_0:
        स्थिती HTP_TYPE_IQ4_NL: परत (ne00 / 32) * 18;
        स्थिती HTP_TYPE_Q4_1:   परत (ne00 / 32) * 20;
        स्थिती HTP_TYPE_MXFP4:  परत (ne00 / 32) * 17;
        स्थिती HTP_TYPE_Q2_K:   परत (ne00 / 256) * 84;
        स्थिती HTP_TYPE_Q2_0:   परत (ne00 / 64) * 18;
        स्थिती HTP_TYPE_Q3_K:   परत (ne00 / 256) * 110;
        स्थिती HTP_TYPE_Q4_K:   परत (ne00 / 256) * 144;
        स्थिती HTP_TYPE_Q5_K:   परत (ne00 / 256) * 176;
        स्थिती HTP_TYPE_Q6_K:   परत (ne00 / 256) * 210;
        स्थिती HTP_TYPE_Q1_0:   परत (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        स्थिती HTP_TYPE_I32:    परत ne00 * 4;
        स्थिती HTP_TYPE_I64:    परत ne00 * 8;
        पूर्वनिर्धारित:              परत 0;
    }
}

#endif // HTP_TENSOR_H
