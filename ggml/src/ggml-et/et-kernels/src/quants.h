// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#अन्तर्भाव "math_fp.h"

#अन्तर्भाव <stdint.h>

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

स्थिर अन्तःस्थ शून्य dequantize_q8_0_block(ध्रुव block_q8_0 * block, भिन्न * dst) {
    ध्रुव भिन्न scale = fp16_to_fp32(block->d);

    प्रति (पूर्णाङ्क i = 0; i < QK8_0; i++) {
        dst[i] = scale * (भिन्न) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
स्थिर अन्तःस्थ शून्य dequantize_q4_0_block(ध्रुव block_q4_0 * block, भिन्न * dst) {
    ध्रुव भिन्न scale = fp16_to_fp32(block->d);

    प्रति (पूर्णाङ्क i = 0; i < QK4_0 / 2; i++) {
        ध्रुव uint8_t byte = block->qs[i];
        dst[i]             = scale * (भिन्न) ((पूर्णाङ्क) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (भिन्न) ((पूर्णाङ्क) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
स्थिर अन्तःस्थ शून्य get_scale_min_k4(पूर्णाङ्क j, ध्रुव uint8_t * q, uint8_t * d, uint8_t * m) {
    जइ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } अण्णहा {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

स्थिर अन्तःस्थ शून्य dequantize_q4_K_block(ध्रुव block_q4_K * block, भिन्न * dst) {
    ध्रुव uint8_t * q   = block->qs;
    ध्रुव भिन्न     d   = fp16_to_fp32(block->d);
    ध्रुव भिन्न     min = fp16_to_fp32(block->dmin);

    पूर्णाङ्क     is = 0;
    uint8_t sc, m;
    प्रति (पूर्णाङ्क j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        ध्रुव भिन्न d1 = d * sc;
        ध्रुव भिन्न m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        ध्रुव भिन्न d2 = d * sc;
        ध्रुव भिन्न m2 = min * m;
        प्रति (पूर्णाङ्क l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        प्रति (पूर्णाङ्क l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
