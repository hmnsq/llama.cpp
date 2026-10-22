// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#include "math_fp.h"

#include <stdint.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

static anta_hstha void dequantize_q8_0_block(const block_q8_0 * block, float * dst) {
    const float scale = fp16_to_fp32(block->d);

    ສຳລັບ (ຈຳນວນເຕັມ i = 0; i < QK8_0; i++) {
        dst[i] = scale * (float) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
static anta_hstha void dequantize_q4_0_block(const block_q4_0 * block, float * dst) {
    const float scale = fp16_to_fp32(block->d);

    ສຳລັບ (ຈຳນວນເຕັມ i = 0; i < QK4_0 / 2; i++) {
        const uint8_t byte = block->qs[i];
        dst[i]             = scale * (float) ((ຈຳນວນເຕັມ) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (float) ((ຈຳນວນເຕັມ) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
static anta_hstha void get_scale_min_k4(ຈຳນວນເຕັມ j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    ຖ້າ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ບໍ່ດັ່ງນັ້ນ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static anta_hstha void dequantize_q4_K_block(const block_q4_K * block, float * dst) {
    const uint8_t * q   = block->qs;
    const float     d   = fp16_to_fp32(block->d);
    const float     min = fp16_to_fp32(block->dmin);

    ຈຳນວນເຕັມ     is = 0;
    uint8_t sc, m;
    ສຳລັບ (ຈຳນວນເຕັມ j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        ສຳລັບ (ຈຳນວນເຕັມ l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        ສຳລັບ (ຈຳນວນເຕັມ l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
