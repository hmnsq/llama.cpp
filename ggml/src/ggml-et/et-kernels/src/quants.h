// Scalar dequantization helpers and ET-side block-size aliases.

#جيڪڏهن_وصف_نه_ٿيل QUANTS_H
#وصف_ڪريو QUANTS_H

#ضمن "math_fp.h"

#ضمن <stdint.h>

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#وصف_ڪريو QK_F16 32
#وصف_ڪريو QK_F32 16

ساكن مضمن فراغ dequantize_q8_0_بلاڪ(ثابت بلاڪ_q8_0 * بلاڪ, عائم * dst) {
    ثابت عائم scale = fp16_منزل_fp32(بلاڪ->d);

    لكل (صحيح i = 0; i < QK8_0; i++) {
        dst[i] = scale * (عائم) بلاڪ->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
ساكن مضمن فراغ dequantize_q4_0_بلاڪ(ثابت بلاڪ_q4_0 * بلاڪ, عائم * dst) {
    ثابت عائم scale = fp16_منزل_fp32(بلاڪ->d);

    لكل (صحيح i = 0; i < QK4_0 / 2; i++) {
        ثابت uint8_t byte = بلاڪ->qs[i];
        dst[i]             = scale * (عائم) ((صحيح) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (عائم) ((صحيح) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
ساكن مضمن فراغ get_scale_min_k4(صحيح j, ثابت uint8_t * q, uint8_t * d, uint8_t * m) {
    إذا (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } وإلا {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

ساكن مضمن فراغ dequantize_q4_K_بلاڪ(ثابت بلاڪ_q4_K * بلاڪ, عائم * dst) {
    ثابت uint8_t * q   = بلاڪ->qs;
    ثابت عائم     d   = fp16_منزل_fp32(بلاڪ->d);
    ثابت عائم     min = fp16_منزل_fp32(بلاڪ->dmin);

    صحيح     is = 0;
    uint8_t sc, m;
    لكل (صحيح j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, بلاڪ->scales, &sc, &m);
        ثابت عائم d1 = d * sc;
        ثابت عائم m1 = min * m;
        get_scale_min_k4(is + 1, بلاڪ->scales, &sc, &m);
        ثابت عائم d2 = d * sc;
        ثابت عائم m2 = min * m;
        لكل (صحيح l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        لكل (صحيح l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#جيڪڏهن_پورو  // QUANTS_H
