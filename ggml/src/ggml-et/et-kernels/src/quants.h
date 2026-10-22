// Scalar dequantization helpers and ET-side block-size aliases.

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ QUANTS_H
#ವ್ಯಾಖ್ಯಾನಿಸು QUANTS_H

#ಅನ್ತರ್ಭಾವ "math_fp.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#ವ್ಯಾಖ್ಯಾನಿಸು QK_F16 32
#ವ್ಯಾಖ್ಯಾನಿಸು QK_F32 16

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dequantize_q8_0_ಖಂಡ(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಖಂಡ, ದಶಮಾಂಶ * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->d);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QK8_0; i++) {
        dst[i] = scale * (ದಶಮಾಂಶ) ಖಂಡ->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dequantize_q4_0_ಖಂಡ(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * ಖಂಡ, ದಶಮಾಂಶ * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->d);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QK4_0 / 2; i++) {
        ಸ್ಥಿರಾಂಕ uint8_t byte = ಖಂಡ->qs[i];
        dst[i]             = scale * (ದಶಮಾಂಶ) ((ಪೂರ್ಣಾಂಕ) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (ದಶಮಾಂಶ) ((ಪೂರ್ಣಾಂಕ) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಂಕ j, ಸ್ಥಿರಾಂಕ uint8_t * q, uint8_t * d, uint8_t * m) {
    ಆದರೆ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ಇಲ್ಲದಿದ್ದರೆ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dequantize_q4_K_ಖಂಡ(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * ಖಂಡ, ದಶಮಾಂಶ * dst) {
    ಸ್ಥಿರಾಂಕ uint8_t * q   = ಖಂಡ->qs;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d   = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->d);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     min = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->dmin);

    ಪೂರ್ಣಾಂಕ     is = 0;
    uint8_t sc, m;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, ಖಂಡ->scales, &sc, &m);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = d * sc;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = min * m;
        get_scale_min_k4(is + 1, ಖಂಡ->scales, &sc, &m);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d2 = d * sc;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m2 = min * m;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#ಆದರೆ_ಅಂತ್ಯ  // QUANTS_H
