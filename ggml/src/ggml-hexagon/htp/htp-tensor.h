#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "hex-bitmap.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_সারি_আকার   128

// Currently HMX requires 32x32 tile formats (do not change)
#define HTP_TENSOR_TILE_N_COLS      32
#define HTP_TENSOR_TILE_N_ROWS      32

#define HTP_TENSOR_TILE_আকার_F16    2048
#define HTP_TENSOR_TILE_আকার_Q4_0   576
#define HTP_TENSOR_TILE_আকার_Q4_1   640
#define HTP_TENSOR_TILE_আকার_Q8_0   1088
#define HTP_TENSOR_TILE_আকার_IQ4_NL 576
#define HTP_TENSOR_TILE_আকার_MXFP4  544
#define HTP_TENSOR_TILE_আকার_Q4_K   640
#define HTP_TENSOR_TILE_আকার_Q5_K   768
#define HTP_TENSOR_TILE_আকার_Q6_K   896
#define HTP_TENSOR_TILE_আকার_Q3_K   512
#define HTP_TENSOR_TILE_আকার_Q2_K   336
#define HTP_TENSOR_TILE_K_Q1_0      128
#define HTP_TENSOR_TILE_আকার_Q1_0   576
#define HTP_TENSOR_TILE_K_Q2_0      64
#define HTP_TENSOR_TILE_আকার_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

কাঠামো htp_tensor_mdev_range {
    uint32_t start;
    uint32_t গুনতি;
};

স্থির অন্তঃরেখ bool htp_tensor_is_extended(ধ্রুবক কাঠামো htp_tensor * t) {
    ফেরত t && (t->তথ্য >> 32) != 0;
}

স্থির অন্তঃরেখ bool htp_tensor_is_contiguous(ধ্রুবক কাঠামো htp_tensor * t, uint32_t type_আকার) {
    uint32_t পরবর্তী_nb = type_আকার;
    যদি (t->ne[0] != 1 && t->nb[0] != পরবর্তী_nb) {
        ফেরত false;
    }
    পরবর্তী_nb *= t->ne[0];
    জন্য (পূর্ণসংখ্যা i = 1; i < HTP_OP_MAX_DIMS; i++) {
        যদি (t->ne[i] != 1 && t->nb[i] != পরবর্তী_nb) {
            ফেরত false;
        }
        পরবর্তী_nb *= t->ne[i];
    }
    ফেরত true;
}

স্থির অন্তঃরেখ bool htp_tensor_is_permuted(ধ্রুবক কাঠামো htp_tensor * t) {
    ফেরত t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

স্থির অন্তঃরেখ bool htp_tensor_mdev_তথ্য_aligned(ধ্রুবক কাঠামো htp_tensor * t) {
    ফেরত ((uintptr_t) t->তথ্য & (HTP_TENSOR_MDEV_সারি_আকার - 1)) == 0;
}

স্থির অন্তঃরেখ bool htp_tensor_can_row_partition(ধ্রুবক কাঠামো htp_tensor * t, uint32_t elem_আকার) {
    যদি (!htp_tensor_mdev_তথ্য_aligned(t)) {
        ফেরত false;
    }
    যদি (t->ne[0] != 1 && t->nb[0] != elem_আকার) {
        ফেরত false;
    }
    যদি (htp_tensor_is_permuted(t)) {
        ফেরত false;
    }
    যদি (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) != 0) ফেরত false;
    যদি (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) != 0) ফেরত false;
    যদি (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) != 0) ফেরত false;
    ফেরত true;
}

স্থির অন্তঃরেখ bool htp_tensor_mdev_rows_per_chunk(ধ্রুবক কাঠামো htp_tensor * t, uint32_t elem_আকার, uint32_t row_আকার, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    যদি (!htp_tensor_mdev_তথ্য_aligned(t)) {
        ফেরত false;
    }
    যদি (t->ne[0] != 1 && t->nb[0] != elem_আকার) {
        ফেরত false;
    }
    যদি (htp_tensor_is_permuted(t)) {
        ফেরত false;
    }
    যদি (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) == 0)) {
        *rows_per_chunk = 1;
        ফেরত true;
    }
    যদি (t->nb[1] == row_আকার &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_আকার > 0) ? (HTP_TENSOR_MDEV_সারি_আকার / hex_gcd_u32(row_আকার, HTP_TENSOR_MDEV_সারি_আকার)) : 1;
        ফেরত true;
    }
    ফেরত false;
}

স্থির অন্তঃরেখ কাঠামো htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t যোগ_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_গুনতি, ধ্রুবক কাঠামো fastdiv_মানসমূহ * mdev_গুনতি_div) {
    কাঠামো htp_tensor_mdev_range range = { 0, যোগ_units };

    যদি (mdev_গুনতি <= 1) {
        ফেরত range;
    }

    যদি (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : যোগ_units;
        range.গুনতি = (mdev_idx == 0) ? যোগ_units : 0;
        ফেরত range;
    }

    ধ্রুবক uint32_t যোগ_chunks = যোগ_units / units_per_chunk;
    যদি (যোগ_chunks < mdev_গুনতি) {
        range.start = (mdev_idx == 0) ? 0 : যোগ_units;
        range.গুনতি = (mdev_idx == 0) ? যোগ_units : 0;
        ফেরত range;
    }

    ধ্রুবক uint32_t chunks_per_mdev = fastdiv(যোগ_chunks + mdev_গুনতি - 1, mdev_গুনতি_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, যোগ_units);
    যদি (mdev_idx == mdev_গুনতি - 1) {
        range.গুনতি = যোগ_units - range.start;
    } অন্যথা {
        range.গুনতি = MIN(chunks_per_mdev * units_per_chunk, যোগ_units - range.start);
    }
    ফেরত range;
}

স্থির অন্তঃরেখ uint32_t htp_tensor_get_tile_আকার(পূর্ণসংখ্যা type) {
    নির্বাচন (type) {
        ক্ষেত্র HTP_TYPE_Q4_0:   ফেরত HTP_TENSOR_TILE_আকার_Q4_0;
        ক্ষেত্র HTP_TYPE_IQ4_NL: ফেরত HTP_TENSOR_TILE_আকার_IQ4_NL;
        ক্ষেত্র HTP_TYPE_Q4_1:   ফেরত HTP_TENSOR_TILE_আকার_Q4_1;
        ক্ষেত্র HTP_TYPE_Q4_K:   ফেরত HTP_TENSOR_TILE_আকার_Q4_K;
        ক্ষেত্র HTP_TYPE_Q8_0:   ফেরত HTP_TENSOR_TILE_আকার_Q8_0;
        ক্ষেত্র HTP_TYPE_Q5_K:   ফেরত HTP_TENSOR_TILE_আকার_Q5_K;
        ক্ষেত্র HTP_TYPE_Q6_K:   ফেরত HTP_TENSOR_TILE_আকার_Q6_K;
        ক্ষেত্র HTP_TYPE_Q3_K:   ফেরত HTP_TENSOR_TILE_আকার_Q3_K;
        ক্ষেত্র HTP_TYPE_Q2_K:   ফেরত HTP_TENSOR_TILE_আকার_Q2_K;
        ক্ষেত্র HTP_TYPE_Q1_0:   ফেরত HTP_TENSOR_TILE_আকার_Q1_0;
        ক্ষেত্র HTP_TYPE_Q2_0:   ফেরত HTP_TENSOR_TILE_আকার_Q2_0;
        ক্ষেত্র HTP_TYPE_MXFP4:  ফেরত HTP_TENSOR_TILE_আকার_MXFP4;
        ক্ষেত্র HTP_TYPE_F16:    ফেরত HTP_TENSOR_TILE_আকার_F16;
        পূর্বনির্ধারিত:              ফেরত 0;
    }
}

স্থির অন্তঃরেখ uint32_t htp_tensor_get_tile_k(পূর্ণসংখ্যা type) {
    নির্বাচন (type) {
        ক্ষেত্র HTP_TYPE_Q1_0: ফেরত HTP_TENSOR_TILE_K_Q1_0;
        ক্ষেত্র HTP_TYPE_Q2_0: ফেরত HTP_TENSOR_TILE_K_Q2_0;
        পূর্বনির্ধারিত:            ফেরত 32;
    }
}

স্থির অন্তঃরেখ uint32_t htp_tensor_get_tile_padded_আকার(পূর্ণসংখ্যা type) {
    ফেরত (htp_tensor_get_tile_আকার(type) + 127u) & ~127u;
}

স্থির অন্তঃরেখ bool htp_tensor_is_tiled(ধ্রুবক কাঠামো htp_tensor * t) {
    ফেরত t && (t->flags & HTP_TENSOR_TILED) != 0;
}

স্থির অন্তঃরেখ uint32_t htp_tensor_get_row_আকার(পূর্ণসংখ্যা type, uint32_t ne00) {
    নির্বাচন (type) {
        ক্ষেত্র HTP_TYPE_F32:    ফেরত ne00 * 4;
        ক্ষেত্র HTP_TYPE_F16:
        ক্ষেত্র HTP_TYPE_BF16:   ফেরত ne00 * 2;
        ক্ষেত্র HTP_TYPE_Q8_0:   ফেরত (ne00 / 32) * 34;
        ক্ষেত্র HTP_TYPE_Q4_0:
        ক্ষেত্র HTP_TYPE_IQ4_NL: ফেরত (ne00 / 32) * 18;
        ক্ষেত্র HTP_TYPE_Q4_1:   ফেরত (ne00 / 32) * 20;
        ক্ষেত্র HTP_TYPE_MXFP4:  ফেরত (ne00 / 32) * 17;
        ক্ষেত্র HTP_TYPE_Q2_K:   ফেরত (ne00 / 256) * 84;
        ক্ষেত্র HTP_TYPE_Q2_0:   ফেরত (ne00 / 64) * 18;
        ক্ষেত্র HTP_TYPE_Q3_K:   ফেরত (ne00 / 256) * 110;
        ক্ষেত্র HTP_TYPE_Q4_K:   ফেরত (ne00 / 256) * 144;
        ক্ষেত্র HTP_TYPE_Q5_K:   ফেরত (ne00 / 256) * 176;
        ক্ষেত্র HTP_TYPE_Q6_K:   ফেরত (ne00 / 256) * 210;
        ক্ষেত্র HTP_TYPE_Q1_0:   ফেরত (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        ক্ষেত্র HTP_TYPE_I32:    ফেরত ne00 * 4;
        ক্ষেত্র HTP_TYPE_I64:    ফেরত ne00 * 8;
        পূর্বনির্ধারিত:              ফেরত 0;
    }
}

#endif // HTP_TENSOR_H
