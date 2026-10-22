// Scalar dequantization helpers and ET-side block-size aliases.

#यदि_अपरिभाषितम् QUANTS_H
#परिभाषय QUANTS_H

#अन्तर्भाव "math_fp.h"

#अन्तर्भाव <stdint.h>

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#परिभाषय QK_F16 32
#परिभाषय QK_F32 16

स्थिर अन्तःस्थ शून्य dequantize_q8_0_खण्डम्(ध्रुव खण्डम्_q8_0 * खण्डम्, भिन्न * dst) {
    ध्रुव भिन्न scale = fp16_गन्तव्यम्_fp32(खण्डम्->d);

    प्रति (पूर्णाङ्क i = 0; i < QK8_0; i++) {
        dst[i] = scale * (भिन्न) खण्डम्->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
स्थिर अन्तःस्थ शून्य dequantize_q4_0_खण्डम्(ध्रुव खण्डम्_q4_0 * खण्डम्, भिन्न * dst) {
    ध्रुव भिन्न scale = fp16_गन्तव्यम्_fp32(खण्डम्->d);

    प्रति (पूर्णाङ्क i = 0; i < QK4_0 / 2; i++) {
        ध्रुव uint8_t byte = खण्डम्->qs[i];
        dst[i]             = scale * (भिन्न) ((पूर्णाङ्क) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (भिन्न) ((पूर्णाङ्क) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
स्थिर अन्तःस्थ शून्य get_scale_min_k4(पूर्णाङ्क j, ध्रुव uint8_t * q, uint8_t * d, uint8_t * m) {
    यदि (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } अन्यथा {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

स्थिर अन्तःस्थ शून्य dequantize_q4_K_खण्डम्(ध्रुव खण्डम्_q4_K * खण्डम्, भिन्न * dst) {
    ध्रुव uint8_t * q   = खण्डम्->qs;
    ध्रुव भिन्न     d   = fp16_गन्तव्यम्_fp32(खण्डम्->d);
    ध्रुव भिन्न     min = fp16_गन्तव्यम्_fp32(खण्डम्->dmin);

    पूर्णाङ्क     is = 0;
    uint8_t sc, m;
    प्रति (पूर्णाङ्क j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, खण्डम्->scales, &sc, &m);
        ध्रुव भिन्न d1 = d * sc;
        ध्रुव भिन्न m1 = min * m;
        get_scale_min_k4(is + 1, खण्डम्->scales, &sc, &m);
        ध्रुव भिन्न d2 = d * sc;
        ध्रुव भिन्न m2 = min * m;
        प्रति (पूर्णाङ्क l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        प्रति (पूर्णाङ्क l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#यदि_अन्तः  // QUANTS_H
