#pragma once

#ಅನ್ತರ್ಭಾವ "common.h"

ಶೂನ್ಯ quantize_q1_0(device ಧ್ರುವ ಭಿನ್ನ * src, device block_q1_0 & dst) {
    ಭಿನ್ನ sum_abs = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK1_0; j++) {
        ಯದಿ (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

ಶೂನ್ಯ quantize_q2_0(device ಧ್ರುವ ಭಿನ್ನ * src, device block_q2_0 & dst) {
    ಭಿನ್ನ amax = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK2_0; j++) {
        ಭಿನ್ನ a = fabs(src[j]);
        ಯದಿ (a > amax) amax = a;
    }
    ಧ್ರುವ ಭಿನ್ನ d = amax;
    dst.d = d;

    ಧ್ರುವ ಭಿನ್ನ id = d > 0.0f ? 1.0f / d : 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK2_0; j++) {
        ಪೂರ್ಣಾಙ್ಕ q = (ಪೂರ್ಣಾಙ್ಕ)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

ಶೂನ್ಯ quantize_q4_0(device ಧ್ರುವ ಭಿನ್ನ * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    ಭಿನ್ನ amax = 0.0f; // absolute max
    ಭಿನ್ನ max  = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_0; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = src[j];
        ಯದಿ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ಧ್ರುವ ಭಿನ್ನ d = max / -8;
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_0/2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = src[0       + j]*id;
        ಧ್ರುವ ಭಿನ್ನ x1 = src[QK4_0/2 + j]*id;

        ಧ್ರುವ uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ಧ್ರುವ uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ಶೂನ್ಯ quantize_q4_1(device ಧ್ರುವ ಭಿನ್ನ * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    ಭಿನ್ನ min = FLT_MAX;
    ಭಿನ್ನ max = -FLT_MAX;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_1; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = src[j];
        ಯದಿ (min > v) min = v;
        ಯದಿ (max < v) max = v;
    }

    ಧ್ರುವ ಭಿನ್ನ d = (max - min) / ((1 << 4) - 1);
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_1/2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = (src[0       + j] - min)*id;
        ಧ್ರುವ ಭಿನ್ನ x1 = (src[QK4_1/2 + j] - min)*id;

        ಧ್ರುವ uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ಧ್ರುವ uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ಶೂನ್ಯ quantize_q5_0(device ಧ್ರುವ ಭಿನ್ನ * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    ಭಿನ್ನ amax = 0.0f; // absolute max
    ಭಿನ್ನ max  = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK5_0; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = src[j];
        ಯದಿ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ಧ್ರುವ ಭಿನ್ನ d = max / -16;
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK5_0/2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = src[0       + j]*id;
        ಧ್ರುವ ಭಿನ್ನ x1 = src[QK5_0/2 + j]*id;

        ಧ್ರುವ uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ಧ್ರುವ uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ಧ್ರುವ uint8_t * qh8 = (thread ಧ್ರುವ uint8_t *)&qh;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ಶೂನ್ಯ quantize_q5_1(device ಧ್ರುವ ಭಿನ್ನ * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    ಭಿನ್ನ max = src[0];
    ಭಿನ್ನ min = src[0];

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 1; j < QK5_1; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ಧ್ರುವ ಭಿನ್ನ d = (max - min) / 31;
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK5_1/2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = (src[0       + j] - min)*id;
        ಧ್ರುವ ಭಿನ್ನ x1 = (src[QK5_1/2 + j] - min)*id;

        ಧ್ರುವ uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ಧ್ರುವ uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ಧ್ರುವ uint8_t * qh8 = (thread ಧ್ರುವ uint8_t *)&qh;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ಶೂನ್ಯ quantize_q8_0(device ಧ್ರುವ ಭಿನ್ನ * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    ಭಿನ್ನ amax = 0.0f; // absolute max

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK8_0; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ಧ್ರುವ ಭಿನ್ನ d = amax / ((1 << 7) - 1);
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK8_0; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

ಶೂನ್ಯ quantize_iq4_nl(device ಧ್ರುವ ಭಿನ್ನ * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    ಭಿನ್ನ amax = 0.0f; // absolute max
    ಭಿನ್ನ max  = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_NL; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = src[j];
        ಯದಿ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ಧ್ರುವ ಭಿನ್ನ d = max / kvalues_iq4nl_f[0];
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;

    ಭಿನ್ನ sumqx = 0, sumq2 = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_NL/2; ++j) {
        ಧ್ರುವ ಭಿನ್ನ x0 = src[0        + j]*id;
        ಧ್ರುವ ಭಿನ್ನ x1 = src[QK4_NL/2 + j]*id;

        ಧ್ರುವ uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        ಧ್ರುವ uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ಧ್ರುವ ಭಿನ್ನ v0 = kvalues_iq4nl_f[xi0];
        ಧ್ರುವ ಭಿನ್ನ v1 = kvalues_iq4nl_f[xi1];
        ಧ್ರುವ ಭಿನ್ನ w0 = src[0        + j]*src[0        + j];
        ಧ್ರುವ ಭಿನ್ನ w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

ಶೂನ್ಯ quantize_tq2_0(device ಧ್ರುವ ಭಿನ್ನ * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    ಭಿನ್ನ amax = 0.0f; // absolute max

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K; j++) {
        ಧ್ರುವ ಭಿನ್ನ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ಧ್ರುವ ಭಿನ್ನ d = amax;
    ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/4; j += 32) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ m = 0; m < 32; ++m) {
            uint8_t q = 0;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                ಪೂರ್ಣಾಙ್ಕ xi = (ಪೂರ್ಣಾಙ್ಕ)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
