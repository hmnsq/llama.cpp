// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#ಅನ್ತರ್ಭಾವ "math_fp.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dequantize_q8_0_block(ಧ್ರುವ block_q8_0 * block, ಭಿನ್ನ * dst) {
    ಧ್ರುವ ಭಿನ್ನ scale = fp16_to_fp32(block->d);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QK8_0; i++) {
        dst[i] = scale * (ಭಿನ್ನ) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dequantize_q4_0_block(ಧ್ರುವ block_q4_0 * block, ಭಿನ್ನ * dst) {
    ಧ್ರುವ ಭಿನ್ನ scale = fp16_to_fp32(block->d);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QK4_0 / 2; i++) {
        ಧ್ರುವ uint8_t byte = block->qs[i];
        dst[i]             = scale * (ಭಿನ್ನ) ((ಪೂರ್ಣಾಙ್ಕ) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (ಭಿನ್ನ) ((ಪೂರ್ಣಾಙ್ಕ) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಙ್ಕ j, ಧ್ರುವ uint8_t * q, uint8_t * d, uint8_t * m) {
    ಯದಿ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ಅನ್ಯಥಾ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dequantize_q4_K_block(ಧ್ರುವ block_q4_K * block, ಭಿನ್ನ * dst) {
    ಧ್ರುವ uint8_t * q   = block->qs;
    ಧ್ರುವ ಭಿನ್ನ     d   = fp16_to_fp32(block->d);
    ಧ್ರುವ ಭಿನ್ನ     min = fp16_to_fp32(block->dmin);

    ಪೂರ್ಣಾಙ್ಕ     is = 0;
    uint8_t sc, m;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        ಧ್ರುವ ಭಿನ್ನ d1 = d * sc;
        ಧ್ರುವ ಭಿನ್ನ m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        ಧ್ರುವ ಭಿನ್ನ d2 = d * sc;
        ಧ್ರುವ ಭಿನ್ನ m2 = min * m;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
