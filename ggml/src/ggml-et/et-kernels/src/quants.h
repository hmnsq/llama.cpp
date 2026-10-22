// Scalar dequantization helpers and ET-side block-size aliases.

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ QUANTS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ QUANTS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ QK_F16 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ QK_F32 16

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dequantize_q8_0_ਖੰਡ(ਅਚਲ ਖੰਡ_q8_0 * ਖੰਡ, ਦਸ਼ਮਲਵ * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ scale = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->d);

    ਲਈ (ਪੂਰਨ i = 0; i < QK8_0; i++) {
        dst[i] = scale * (ਦਸ਼ਮਲਵ) ਖੰਡ->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dequantize_q4_0_ਖੰਡ(ਅਚਲ ਖੰਡ_q4_0 * ਖੰਡ, ਦਸ਼ਮਲਵ * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ scale = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->d);

    ਲਈ (ਪੂਰਨ i = 0; i < QK4_0 / 2; i++) {
        ਅਚਲ uint8_t byte = ਖੰਡ->qs[i];
        dst[i]             = scale * (ਦਸ਼ਮਲਵ) ((ਪੂਰਨ) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (ਦਸ਼ਮਲਵ) ((ਪੂਰਨ) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ get_scale_min_k4(ਪੂਰਨ j, ਅਚਲ uint8_t * q, uint8_t * d, uint8_t * m) {
    ਜੇ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ਨਹੀਂਤਾਂ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dequantize_q4_K_ਖੰਡ(ਅਚਲ ਖੰਡ_q4_K * ਖੰਡ, ਦਸ਼ਮਲਵ * dst) {
    ਅਚਲ uint8_t * q   = ਖੰਡ->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ     d   = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->d);
    ਅਚਲ ਦਸ਼ਮਲਵ     min = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->dmin);

    ਪੂਰਨ     is = 0;
    uint8_t sc, m;
    ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, ਖੰਡ->scales, &sc, &m);
        ਅਚਲ ਦਸ਼ਮਲਵ d1 = d * sc;
        ਅਚਲ ਦਸ਼ਮਲਵ m1 = min * m;
        get_scale_min_k4(is + 1, ਖੰਡ->scales, &sc, &m);
        ਅਚਲ ਦਸ਼ਮਲਵ d2 = d * sc;
        ਅਚਲ ਦਸ਼ਮਲਵ m2 = min * m;
        ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#ਜੇ_ਅੰਤ  // QUANTS_H
