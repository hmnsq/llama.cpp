// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#અન્તર્ભાવ "math_fp.h"

#અન્તર્ભાવ <stdint.h>

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

સ્થિર અંતઃરેખ શૂન્ય dequantize_q8_0_block(અચલ block_q8_0 * block, દશાંશ * dst) {
    અચલ દશાંશ scale = fp16_to_fp32(block->d);

    માટે (પૂર્ણાંક i = 0; i < QK8_0; i++) {
        dst[i] = scale * (દશાંશ) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
સ્થિર અંતઃરેખ શૂન્ય dequantize_q4_0_block(અચલ block_q4_0 * block, દશાંશ * dst) {
    અચલ દશાંશ scale = fp16_to_fp32(block->d);

    માટે (પૂર્ણાંક i = 0; i < QK4_0 / 2; i++) {
        અચલ uint8_t byte = block->qs[i];
        dst[i]             = scale * (દશાંશ) ((પૂર્ણાંક) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (દશાંશ) ((પૂર્ણાંક) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
સ્થિર અંતઃરેખ શૂન્ય get_scale_min_k4(પૂર્ણાંક j, અચલ uint8_t * q, uint8_t * d, uint8_t * m) {
    જો (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } નહીંતર {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

સ્થિર અંતઃરેખ શૂન્ય dequantize_q4_K_block(અચલ block_q4_K * block, દશાંશ * dst) {
    અચલ uint8_t * q   = block->qs;
    અચલ દશાંશ     d   = fp16_to_fp32(block->d);
    અચલ દશાંશ     min = fp16_to_fp32(block->dmin);

    પૂર્ણાંક     is = 0;
    uint8_t sc, m;
    માટે (પૂર્ણાંક j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        અચલ દશાંશ d1 = d * sc;
        અચલ દશાંશ m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        અચલ દશાંશ d2 = d * sc;
        અચલ દશાંશ m2 = min * m;
        માટે (પૂર્ણાંક l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        માટે (પૂર્ણાંક l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
