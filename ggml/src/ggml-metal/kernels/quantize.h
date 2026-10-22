#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "common.h"

ಶೂನ್ಯ quantize_q1_0(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_q1_0 & dst) {
    ದಶಮಾಂಶ ಮೊತ್ತ_abs = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK1_0; j++) {
        ಮೊತ್ತ_abs += fabs(src[j]);
    }
    dst.d = ಮೊತ್ತ_abs / QK1_0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK1_0; j++) {
        ಆದರೆ (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

ಶೂನ್ಯ quantize_q2_0(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_q2_0 & dst) {
    ದಶಮಾಂಶ amax = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK2_0; j++) {
        ದಶಮಾಂಶ a = fabs(src[j]);
        ಆದರೆ (a > amax) amax = a;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax;
    dst.d = d;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d > 0.0f ? 1.0f / d : 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK2_0; j++) {
        ಪೂರ್ಣಾಂಕ q = (ಪೂರ್ಣಾಂಕ)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

ಶೂನ್ಯ quantize_q4_0(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_q4_0 & dst) {
#ಸೂಚನೆ METAL fp math_ಬಹುಲಕ(ಸುರಕ್ಷಿತ)
    ದಶಮಾಂಶ amax = 0.0f; // absolute max
    ದಶಮಾಂಶ max  = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = src[j];
        ಆದರೆ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = max / -8;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0/2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = src[0       + j]*id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = src[QK4_0/2 + j]*id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ಶೂನ್ಯ quantize_q4_1(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_q4_1 & dst) {
#ಸೂಚನೆ METAL fp math_ಬಹುಲಕ(ಸುರಕ್ಷಿತ)
    ದಶಮಾಂಶ min = FLT_MAX;
    ದಶಮಾಂಶ max = -FLT_MAX;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = src[j];
        ಆದರೆ (min > v) min = v;
        ಆದರೆ (max < v) max = v;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (max - min) / ((1 << 4) - 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1/2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = (src[0       + j] - min)*id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = (src[QK4_1/2 + j] - min)*id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ಶೂನ್ಯ quantize_q5_0(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_q5_0 & dst) {
#ಸೂಚನೆ METAL fp math_ಬಹುಲಕ(ಸುರಕ್ಷಿತ)
    ದಶಮಾಂಶ amax = 0.0f; // absolute max
    ದಶಮಾಂಶ max  = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_0; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = src[j];
        ಆದರೆ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = max / -16;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_0/2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = src[0       + j]*id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = src[QK5_0/2 + j]*id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ಸ್ಥಿರಾಂಕ uint8_t * qh8 = (thread ಸ್ಥಿರಾಂಕ uint8_t *)&qh;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ಶೂನ್ಯ quantize_q5_1(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_q5_1 & dst) {
#ಸೂಚನೆ METAL fp math_ಬಹುಲಕ(ಸುರಕ್ಷಿತ)
    ದಶಮಾಂಶ max = src[0];
    ದಶಮಾಂಶ min = src[0];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < QK5_1; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (max - min) / 31;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_1/2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = (src[0       + j] - min)*id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = (src[QK5_1/2 + j] - min)*id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ಸ್ಥಿರಾಂಕ uint8_t * qh8 = (thread ಸ್ಥಿರಾಂಕ uint8_t *)&qh;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ಶೂನ್ಯ quantize_q8_0(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_q8_0 & dst) {
#ಸೂಚನೆ METAL fp math_ಬಹುಲಕ(ಸುರಕ್ಷಿತ)
    ದಶಮಾಂಶ amax = 0.0f; // absolute max

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK8_0; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / ((1 << 7) - 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK8_0; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

ಶೂನ್ಯ quantize_iq4_nl(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_iq4_nl & dst) {
#ಸೂಚನೆ METAL fp math_ಬಹುಲಕ(ಸುರಕ್ಷಿತ)
    ದಶಮಾಂಶ amax = 0.0f; // absolute max
    ದಶಮಾಂಶ max  = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_NL; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = src[j];
        ಆದರೆ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = max / kvalues_iq4nl_f[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

    ದಶಮಾಂಶ sumqx = 0, sumq2 = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_NL/2; ++j) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = src[0        + j]*id;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = src[QK4_NL/2 + j]*id;

        ಸ್ಥಿರಾಂಕ uint8_t xi0 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_int8(16, kvalues_iq4nl_f, x0);
        ಸ್ಥಿರಾಂಕ uint8_t xi1 = ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v0 = kvalues_iq4nl_f[xi0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v1 = kvalues_iq4nl_f[xi1];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w0 = src[0        + j]*src[0        + j];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

ಶೂನ್ಯ quantize_tq2_0(device ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, device ಖಂಡ_tq2_0 & dst) {
#ಸೂಚನೆ METAL fp math_ಬಹುಲಕ(ಸುರಕ್ಷಿತ)
    ದಶಮಾಂಶ amax = 0.0f; // absolute max

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/4; j += 32) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m = 0; m < 32; ++m) {
            uint8_t q = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                ಪೂರ್ಣಾಂಕ xi = (ಪೂರ್ಣಾಂಕ)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
