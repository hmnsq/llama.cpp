// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#شامل "math_fp.h"

#شامل <stdint.h>

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

ایستا درونخطی تهی dequantize_q8_0_block(ثابت block_q8_0 * block, اعشاری * dst) {
    ثابت اعشاری scale = fp16_to_fp32(block->d);

    برای (صحیح i = 0; i < QK8_0; i++) {
        dst[i] = scale * (اعشاری) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
ایستا درونخطی تهی dequantize_q4_0_block(ثابت block_q4_0 * block, اعشاری * dst) {
    ثابت اعشاری scale = fp16_to_fp32(block->d);

    برای (صحیح i = 0; i < QK4_0 / 2; i++) {
        ثابت uint8_t byte = block->qs[i];
        dst[i]             = scale * (اعشاری) ((صحیح) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (اعشاری) ((صحیح) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
ایستا درونخطی تهی get_scale_min_k4(صحیح j, ثابت uint8_t * q, uint8_t * d, uint8_t * m) {
    اگر (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } وگرنه {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

ایستا درونخطی تهی dequantize_q4_K_block(ثابت block_q4_K * block, اعشاری * dst) {
    ثابت uint8_t * q   = block->qs;
    ثابت اعشاری     d   = fp16_to_fp32(block->d);
    ثابت اعشاری     min = fp16_to_fp32(block->dmin);

    صحیح     is = 0;
    uint8_t sc, m;
    برای (صحیح j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        ثابت اعشاری d1 = d * sc;
        ثابت اعشاری m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        ثابت اعشاری d2 = d * sc;
        ثابت اعشاری m2 = min * m;
        برای (صحیح l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        برای (صحیح l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
