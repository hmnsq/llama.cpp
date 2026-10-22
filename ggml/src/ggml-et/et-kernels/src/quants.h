// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#অন্তর্ভাব "math_fp.h"

#অন্তর্ভাব <stdint.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

স্থির অন্তঃরেখ শূন্য dequantize_q8_0_block(ধ্রুবক block_q8_0 * block, দশমিক * dst) {
    ধ্রুবক দশমিক scale = fp16_to_fp32(block->d);

    জন্য (পূর্ণসংখ্যা i = 0; i < QK8_0; i++) {
        dst[i] = scale * (দশমিক) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
স্থির অন্তঃরেখ শূন্য dequantize_q4_0_block(ধ্রুবক block_q4_0 * block, দশমিক * dst) {
    ধ্রুবক দশমিক scale = fp16_to_fp32(block->d);

    জন্য (পূর্ণসংখ্যা i = 0; i < QK4_0 / 2; i++) {
        ধ্রুবক uint8_t byte = block->qs[i];
        dst[i]             = scale * (দশমিক) ((পূর্ণসংখ্যা) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (দশমিক) ((পূর্ণসংখ্যা) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
স্থির অন্তঃরেখ শূন্য get_scale_min_k4(পূর্ণসংখ্যা j, ধ্রুবক uint8_t * q, uint8_t * d, uint8_t * m) {
    যদি (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } অন্যথা {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

স্থির অন্তঃরেখ শূন্য dequantize_q4_K_block(ধ্রুবক block_q4_K * block, দশমিক * dst) {
    ধ্রুবক uint8_t * q   = block->qs;
    ধ্রুবক দশমিক     d   = fp16_to_fp32(block->d);
    ধ্রুবক দশমিক     min = fp16_to_fp32(block->dmin);

    পূর্ণসংখ্যা     is = 0;
    uint8_t sc, m;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        ধ্রুবক দশমিক d1 = d * sc;
        ধ্রুবক দশমিক m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        ধ্রুবক দশমিক d2 = d * sc;
        ধ্রুবক দশমিক m2 = min * m;
        জন্য (পূর্ণসংখ্যা l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        জন্য (পূর্ণসংখ্যা l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
