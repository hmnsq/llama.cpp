// Scalar dequantization helpers and ET-side block-size aliases.

#اگر_غیر_تعریف_شدہ QUANTS_H
#تعریف_کرو QUANTS_H

#شامل "math_fp.h"

#شامل <stdint.h>

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#تعریف_کرو QK_F16 32
#تعریف_کرو QK_F32 16

ساکن اندرونی خالی dequantize_q8_0_بلاک(ثابت بلاک_q8_0 * بلاک, اعشاری * dst) {
    ثابت اعشاری scale = fp16_مقصد_fp32(بلاک->d);

    برائے (صحیح i = 0; i < QK8_0; i++) {
        dst[i] = scale * (اعشاری) بلاک->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
ساکن اندرونی خالی dequantize_q4_0_بلاک(ثابت بلاک_q4_0 * بلاک, اعشاری * dst) {
    ثابت اعشاری scale = fp16_مقصد_fp32(بلاک->d);

    برائے (صحیح i = 0; i < QK4_0 / 2; i++) {
        ثابت uint8_t byte = بلاک->qs[i];
        dst[i]             = scale * (اعشاری) ((صحیح) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (اعشاری) ((صحیح) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
ساکن اندرونی خالی get_scale_min_k4(صحیح j, ثابت uint8_t * q, uint8_t * d, uint8_t * m) {
    اگر (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ورنہ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

ساکن اندرونی خالی dequantize_q4_K_بلاک(ثابت بلاک_q4_K * بلاک, اعشاری * dst) {
    ثابت uint8_t * q   = بلاک->qs;
    ثابت اعشاری     d   = fp16_مقصد_fp32(بلاک->d);
    ثابت اعشاری     min = fp16_مقصد_fp32(بلاک->dmin);

    صحیح     is = 0;
    uint8_t sc, m;
    برائے (صحیح j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, بلاک->scales, &sc, &m);
        ثابت اعشاری d1 = d * sc;
        ثابت اعشاری m1 = min * m;
        get_scale_min_k4(is + 1, بلاک->scales, &sc, &m);
        ثابت اعشاری d2 = d * sc;
        ثابت اعشاری m2 = min * m;
        برائے (صحیح l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        برائے (صحیح l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#اگر_ختم  // QUANTS_H
