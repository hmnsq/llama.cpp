#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_TENSOR_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "hex-bitmap.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ   128

// Currently HMX requires 32x32 tile formats (do not change)
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ      32
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_N_ಅಡ್ಡಸಾಲುಗಳು      32

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_F16    2048
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_0   576
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_1   640
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q8_0   1088
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_IQ4_NL 576
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_MXFP4  544
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_K   640
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q5_K   768
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q6_K   896
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q3_K   512
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_K   336
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_K_Q1_0      128
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q1_0   576
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_K_Q2_0      64
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

ರಚನೆ htp_tensor_mdev_range {
    uint32_t ಆರಂಭ;
    uint32_t ಎಣಿಕೆ_2;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_tensor_is_extended(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t) {
    ಹಿಂತಿರುಗಿ t && (t->ದತ್ತಾಂಶ >> 32) != 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_tensor_is_contiguous(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t, uint32_t type_ಗಾತ್ರ) {
    uint32_t ಮುಂದಿನ_nb = type_ಗಾತ್ರ;
    ಆದರೆ (t->ne[0] != 1 && t->nb[0] != ಮುಂದಿನ_nb) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಮುಂದಿನ_nb *= t->ne[0];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < HTP_OP_MAX_DIMS; i++) {
        ಆದರೆ (t->ne[i] != 1 && t->nb[i] != ಮುಂದಿನ_nb) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಮುಂದಿನ_nb *= t->ne[i];
    }
    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_tensor_is_permuted(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t) {
    ಹಿಂತಿರುಗಿ t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t) {
    ಹಿಂತಿರುಗಿ ((uintptr_t) t->ದತ್ತಾಂಶ & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t, uint32_t elem_ಗಾತ್ರ) {
    ಆದರೆ (!htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(t)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (t->ne[0] != 1 && t->nb[0] != elem_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (htp_tensor_is_permuted(t)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) != 0) ಹಿಂತಿರುಗಿ false;
    ಆದರೆ (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) != 0) ಹಿಂತಿರುಗಿ false;
    ಆದರೆ (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) != 0) ಹಿಂತಿರುಗಿ false;
    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t, uint32_t elem_ಗಾತ್ರ, uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, uint32_t * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk) {
    *ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;

    ಆದರೆ (!htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(t)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (t->ne[0] != 1 && t->nb[0] != elem_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (htp_tensor_is_permuted(t)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0)) {
        *ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 1;
        ಹಿಂತಿರುಗಿ true;
    }
    ಆದರೆ (t->nb[1] == ಅಡ್ಡಸಾಲು_ಗಾತ್ರ &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ > 0) ? (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ / hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ)) : 1;
        ಹಿಂತಿರುಗಿ true;
    }
    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ htp_tensor_mdev_range htp_tensor_mdev_ವಿಭಜನೆ(uint32_t ಒಟ್ಟು_units, uint32_t units_ಪ್ರತಿ_chunk, uint32_t mdev_idx, uint32_t mdev_ಎಣಿಕೆ, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * mdev_ಎಣಿಕೆ_div) {
    ರಚನೆ htp_tensor_mdev_range range = { 0, ಒಟ್ಟು_units };

    ಆದರೆ (mdev_ಎಣಿಕೆ <= 1) {
        ಹಿಂತಿರುಗಿ range;
    }

    ಆದರೆ (units_ಪ್ರತಿ_chunk == 0) {
        range.ಆರಂಭ = (mdev_idx == 0) ? 0 : ಒಟ್ಟು_units;
        range.ಎಣಿಕೆ_2 = (mdev_idx == 0) ? ಒಟ್ಟು_units : 0;
        ಹಿಂತಿರುಗಿ range;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_chunks = ಒಟ್ಟು_units / units_ಪ್ರತಿ_chunk;
    ಆದರೆ (ಒಟ್ಟು_chunks < mdev_ಎಣಿಕೆ) {
        range.ಆರಂಭ = (mdev_idx == 0) ? 0 : ಒಟ್ಟು_units;
        range.ಎಣಿಕೆ_2 = (mdev_idx == 0) ? ಒಟ್ಟು_units : 0;
        ಹಿಂತಿರುಗಿ range;
    }

    ಸ್ಥಿರಾಂಕ uint32_t chunks_ಪ್ರತಿ_mdev = fastdiv(ಒಟ್ಟು_chunks + mdev_ಎಣಿಕೆ - 1, mdev_ಎಣಿಕೆ_div);
    range.ಆರಂಭ = MIN(mdev_idx * chunks_ಪ್ರತಿ_mdev * units_ಪ್ರತಿ_chunk, ಒಟ್ಟು_units);
    ಆದರೆ (mdev_idx == mdev_ಎಣಿಕೆ - 1) {
        range.ಎಣಿಕೆ_2 = ಒಟ್ಟು_units - range.ಆರಂಭ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        range.ಎಣಿಕೆ_2 = MIN(chunks_ಪ್ರತಿ_mdev * units_ಪ್ರತಿ_chunk, ಒಟ್ಟು_units - range.ಆರಂಭ);
    }
    ಹಿಂತಿರುಗಿ range;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t htp_tensor_get_tile_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ HTP_TYPE_Q4_0:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_0;
        ಸ್ಥಿತಿ HTP_TYPE_IQ4_NL: ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_IQ4_NL;
        ಸ್ಥಿತಿ HTP_TYPE_Q4_1:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_1;
        ಸ್ಥಿತಿ HTP_TYPE_Q4_K:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_K;
        ಸ್ಥಿತಿ HTP_TYPE_Q8_0:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q8_0;
        ಸ್ಥಿತಿ HTP_TYPE_Q5_K:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q5_K;
        ಸ್ಥಿತಿ HTP_TYPE_Q6_K:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q6_K;
        ಸ್ಥಿತಿ HTP_TYPE_Q3_K:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q3_K;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_K:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_K;
        ಸ್ಥಿತಿ HTP_TYPE_Q1_0:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q1_0;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_0:   ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0;
        ಸ್ಥಿತಿ HTP_TYPE_MXFP4:  ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_MXFP4;
        ಸ್ಥಿತಿ HTP_TYPE_F16:    ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_ಗಾತ್ರ_F16;
        ಪೂರ್ವನಿಯೋಜಿತ:              ಹಿಂತಿರುಗಿ 0;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t htp_tensor_get_tile_k(ಪೂರ್ಣಾಂಕ type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ HTP_TYPE_Q1_0: ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_K_Q1_0;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_0: ಹಿಂತಿರುಗಿ HTP_TENSOR_TILE_K_Q2_0;
        ಪೂರ್ವನಿಯೋಜಿತ:            ಹಿಂತಿರುಗಿ 32;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t htp_tensor_get_tile_padded_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ type) {
    ಹಿಂತಿರುಗಿ (htp_tensor_get_tile_ಗಾತ್ರ(type) + 127u) & ~127u;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_tensor_is_tiled(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t) {
    ಹಿಂತಿರುಗಿ t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ type, uint32_t ne00) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:    ಹಿಂತಿರುಗಿ ne00 * 4;
        ಸ್ಥಿತಿ HTP_TYPE_F16:
        ಸ್ಥಿತಿ HTP_TYPE_BF16:   ಹಿಂತಿರುಗಿ ne00 * 2;
        ಸ್ಥಿತಿ HTP_TYPE_Q8_0:   ಹಿಂತಿರುಗಿ (ne00 / 32) * 34;
        ಸ್ಥಿತಿ HTP_TYPE_Q4_0:
        ಸ್ಥಿತಿ HTP_TYPE_IQ4_NL: ಹಿಂತಿರುಗಿ (ne00 / 32) * 18;
        ಸ್ಥಿತಿ HTP_TYPE_Q4_1:   ಹಿಂತಿರುಗಿ (ne00 / 32) * 20;
        ಸ್ಥಿತಿ HTP_TYPE_MXFP4:  ಹಿಂತಿರುಗಿ (ne00 / 32) * 17;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_K:   ಹಿಂತಿರುಗಿ (ne00 / 256) * 84;
        ಸ್ಥಿತಿ HTP_TYPE_Q2_0:   ಹಿಂತಿರುಗಿ (ne00 / 64) * 18;
        ಸ್ಥಿತಿ HTP_TYPE_Q3_K:   ಹಿಂತಿರುಗಿ (ne00 / 256) * 110;
        ಸ್ಥಿತಿ HTP_TYPE_Q4_K:   ಹಿಂತಿರುಗಿ (ne00 / 256) * 144;
        ಸ್ಥಿತಿ HTP_TYPE_Q5_K:   ಹಿಂತಿರುಗಿ (ne00 / 256) * 176;
        ಸ್ಥಿತಿ HTP_TYPE_Q6_K:   ಹಿಂತಿರುಗಿ (ne00 / 256) * 210;
        ಸ್ಥಿತಿ HTP_TYPE_Q1_0:   ಹಿಂತಿರುಗಿ (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        ಸ್ಥಿತಿ HTP_TYPE_I32:    ಹಿಂತಿರುಗಿ ne00 * 4;
        ಸ್ಥಿತಿ HTP_TYPE_I64:    ಹಿಂತಿರುಗಿ ne00 * 8;
        ಪೂರ್ವನಿಯೋಜಿತ:              ಹಿಂತಿರುಗಿ 0;
    }
}

#ಆದರೆ_ಅಂತ್ಯ // HTP_TENSOR_H
