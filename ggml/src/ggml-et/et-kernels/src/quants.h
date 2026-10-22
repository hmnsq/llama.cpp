// Scalar dequantization helpers and ET-side block-size aliases.

#अगर_अघोषित QUANTS_H
#घोषणा QUANTS_H

#समावेश "math_fp.h"

#समावेश <stdint.h>

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#घोषणा QK_F16 32
#घोषणा QK_F32 16

अटल अंतरभूत व्योम dequantize_q8_0_block(स्थिर block_q8_0 * block, भग्न * dst) {
    स्थिर भग्न scale = fp16_to_fp32(block->d);

    क्रम (पूर्णांक i = 0; i < QK8_0; i++) {
        dst[i] = scale * (भग्न) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
अटल अंतरभूत व्योम dequantize_q4_0_block(स्थिर block_q4_0 * block, भग्न * dst) {
    स्थिर भग्न scale = fp16_to_fp32(block->d);

    क्रम (पूर्णांक i = 0; i < QK4_0 / 2; i++) {
        स्थिर uint8_t byte = block->qs[i];
        dst[i]             = scale * (भग्न) ((पूर्णांक) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (भग्न) ((पूर्णांक) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
अटल अंतरभूत व्योम get_scale_min_k4(पूर्णांक j, स्थिर uint8_t * q, uint8_t * d, uint8_t * m) {
    अगर (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } अन्यथा {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

अटल अंतरभूत व्योम dequantize_q4_K_block(स्थिर block_q4_K * block, भग्न * dst) {
    स्थिर uint8_t * q   = block->qs;
    स्थिर भग्न     d   = fp16_to_fp32(block->d);
    स्थिर भग्न     min = fp16_to_fp32(block->dmin);

    पूर्णांक     is = 0;
    uint8_t sc, m;
    क्रम (पूर्णांक j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        स्थिर भग्न d1 = d * sc;
        स्थिर भग्न m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        स्थिर भग्न d2 = d * sc;
        स्थिर भग्न m2 = min * m;
        क्रम (पूर्णांक l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        क्रम (पूर्णांक l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#पूर्ण_अगर  // QUANTS_H
