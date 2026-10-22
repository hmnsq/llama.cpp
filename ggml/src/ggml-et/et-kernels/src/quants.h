// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#ଅନ୍ତର୍ଭାଵ "math_fp.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dequantize_q8_0_block(ଅଚଳ block_q8_0 * block, ଦଶମିକ * dst) {
    ଅଚଳ ଦଶମିକ scale = fp16_to_fp32(block->d);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < QK8_0; i++) {
        dst[i] = scale * (ଦଶମିକ) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dequantize_q4_0_block(ଅଚଳ block_q4_0 * block, ଦଶମିକ * dst) {
    ଅଚଳ ଦଶମିକ scale = fp16_to_fp32(block->d);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < QK4_0 / 2; i++) {
        ଅଚଳ uint8_t byte = block->qs[i];
        dst[i]             = scale * (ଦଶମିକ) ((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (ଦଶମିକ) ((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ get_scale_min_k4(ପୂର୍ଣ୍ଣସଂଖ୍ୟା j, ଅଚଳ uint8_t * q, uint8_t * d, uint8_t * m) {
    ଯଦି (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ଅନ୍ୟଥା {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dequantize_q4_K_block(ଅଚଳ block_q4_K * block, ଦଶମିକ * dst) {
    ଅଚଳ uint8_t * q   = block->qs;
    ଅଚଳ ଦଶମିକ     d   = fp16_to_fp32(block->d);
    ଅଚଳ ଦଶମିକ     min = fp16_to_fp32(block->dmin);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା     is = 0;
    uint8_t sc, m;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        ଅଚଳ ଦଶମିକ d1 = d * sc;
        ଅଚଳ ଦଶମିକ m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        ଅଚଳ ଦଶମିକ d2 = d * sc;
        ଅଚଳ ଦଶମିକ m2 = min * m;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
