// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#அந்தர்பாவ "math_fp.h"

#அந்தர்பாவ <stdint.h>

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

நிலையான உள்வரி வெறுமை dequantize_q8_0_block(மாறிலி block_q8_0 * block, தசம * dst) {
    மாறிலி தசம scale = fp16_to_fp32(block->d);

    க்கு (முழு i = 0; i < QK8_0; i++) {
        dst[i] = scale * (தசம) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
நிலையான உள்வரி வெறுமை dequantize_q4_0_block(மாறிலி block_q4_0 * block, தசம * dst) {
    மாறிலி தசம scale = fp16_to_fp32(block->d);

    க்கு (முழு i = 0; i < QK4_0 / 2; i++) {
        மாறிலி uint8_t byte = block->qs[i];
        dst[i]             = scale * (தசம) ((முழு) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (தசம) ((முழு) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
நிலையான உள்வரி வெறுமை get_scale_min_k4(முழு j, மாறிலி uint8_t * q, uint8_t * d, uint8_t * m) {
    எனில் (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } இல்லையெனில் {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

நிலையான உள்வரி வெறுமை dequantize_q4_K_block(மாறிலி block_q4_K * block, தசம * dst) {
    மாறிலி uint8_t * q   = block->qs;
    மாறிலி தசம     d   = fp16_to_fp32(block->d);
    மாறிலி தசம     min = fp16_to_fp32(block->dmin);

    முழு     is = 0;
    uint8_t sc, m;
    க்கு (முழு j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        மாறிலி தசம d1 = d * sc;
        மாறிலி தசம m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        மாறிலி தசம d2 = d * sc;
        மாறிலி தசம m2 = min * m;
        க்கு (முழு l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        க்கு (முழு l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
