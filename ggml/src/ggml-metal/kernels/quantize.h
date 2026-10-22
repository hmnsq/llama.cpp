#pragma once

#ଅନ୍ତର୍ଭାଵ "common.h"

ଶୂନ୍ୟ quantize_q1_0(device ଅଚଳ ଦଶମିକ * src, device block_q1_0 & dst) {
    ଦଶମିକ sum_abs = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK1_0; j++) {
        ଯଦି (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

ଶୂନ୍ୟ quantize_q2_0(device ଅଚଳ ଦଶମିକ * src, device block_q2_0 & dst) {
    ଦଶମିକ amax = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK2_0; j++) {
        ଦଶମିକ a = fabs(src[j]);
        ଯଦି (a > amax) amax = a;
    }
    ଅଚଳ ଦଶମିକ d = amax;
    dst.d = d;

    ଅଚଳ ଦଶମିକ id = d > 0.0f ? 1.0f / d : 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK2_0; j++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା q = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

ଶୂନ୍ୟ quantize_q4_0(device ଅଚଳ ଦଶମିକ * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    ଦଶମିକ amax = 0.0f; // absolute max
    ଦଶମିକ max  = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_0; j++) {
        ଅଚଳ ଦଶମିକ v = src[j];
        ଯଦି (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ଅଚଳ ଦଶମିକ d = max / -8;
    ଅଚଳ ଦଶମିକ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_0/2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = src[0       + j]*id;
        ଅଚଳ ଦଶମିକ x1 = src[QK4_0/2 + j]*id;

        ଅଚଳ uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ଅଚଳ uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ଶୂନ୍ୟ quantize_q4_1(device ଅଚଳ ଦଶମିକ * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    ଦଶମିକ min = FLT_MAX;
    ଦଶମିକ max = -FLT_MAX;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_1; j++) {
        ଅଚଳ ଦଶମିକ v = src[j];
        ଯଦି (min > v) min = v;
        ଯଦି (max < v) max = v;
    }

    ଅଚଳ ଦଶମିକ d = (max - min) / ((1 << 4) - 1);
    ଅଚଳ ଦଶମିକ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_1/2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = (src[0       + j] - min)*id;
        ଅଚଳ ଦଶମିକ x1 = (src[QK4_1/2 + j] - min)*id;

        ଅଚଳ uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ଅଚଳ uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ଶୂନ୍ୟ quantize_q5_0(device ଅଚଳ ଦଶମିକ * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    ଦଶମିକ amax = 0.0f; // absolute max
    ଦଶମିକ max  = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK5_0; j++) {
        ଅଚଳ ଦଶମିକ v = src[j];
        ଯଦି (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ଅଚଳ ଦଶମିକ d = max / -16;
    ଅଚଳ ଦଶମିକ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK5_0/2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = src[0       + j]*id;
        ଅଚଳ ଦଶମିକ x1 = src[QK5_0/2 + j]*id;

        ଅଚଳ uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ଅଚଳ uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ଅଚଳ uint8_t * qh8 = (thread ଅଚଳ uint8_t *)&qh;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ଶୂନ୍ୟ quantize_q5_1(device ଅଚଳ ଦଶମିକ * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    ଦଶମିକ max = src[0];
    ଦଶମିକ min = src[0];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < QK5_1; j++) {
        ଅଚଳ ଦଶମିକ v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ଅଚଳ ଦଶମିକ d = (max - min) / 31;
    ଅଚଳ ଦଶମିକ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK5_1/2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = (src[0       + j] - min)*id;
        ଅଚଳ ଦଶମିକ x1 = (src[QK5_1/2 + j] - min)*id;

        ଅଚଳ uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ଅଚଳ uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ଅଚଳ uint8_t * qh8 = (thread ଅଚଳ uint8_t *)&qh;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ଶୂନ୍ୟ quantize_q8_0(device ଅଚଳ ଦଶମିକ * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    ଦଶମିକ amax = 0.0f; // absolute max

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK8_0; j++) {
        ଅଚଳ ଦଶମିକ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ଅଚଳ ଦଶମିକ d = amax / ((1 << 7) - 1);
    ଅଚଳ ଦଶମିକ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK8_0; ++j) {
        ଅଚଳ ଦଶମିକ x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

ଶୂନ୍ୟ quantize_iq4_nl(device ଅଚଳ ଦଶମିକ * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    ଦଶମିକ amax = 0.0f; // absolute max
    ଦଶମିକ max  = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_NL; j++) {
        ଅଚଳ ଦଶମିକ v = src[j];
        ଯଦି (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ଅଚଳ ଦଶମିକ d = max / kvalues_iq4nl_f[0];
    ଅଚଳ ଦଶମିକ id = d ? 1.0f/d : 0.0f;

    ଦଶମିକ sumqx = 0, sumq2 = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_NL/2; ++j) {
        ଅଚଳ ଦଶମିକ x0 = src[0        + j]*id;
        ଅଚଳ ଦଶମିକ x1 = src[QK4_NL/2 + j]*id;

        ଅଚଳ uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        ଅଚଳ uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ଅଚଳ ଦଶମିକ v0 = kvalues_iq4nl_f[xi0];
        ଅଚଳ ଦଶମିକ v1 = kvalues_iq4nl_f[xi1];
        ଅଚଳ ଦଶମିକ w0 = src[0        + j]*src[0        + j];
        ଅଚଳ ଦଶମିକ w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

ଶୂନ୍ୟ quantize_tq2_0(device ଅଚଳ ଦଶମିକ * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    ଦଶମିକ amax = 0.0f; // absolute max

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K; j++) {
        ଅଚଳ ଦଶମିକ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ଅଚଳ ଦଶମିକ d = amax;
    ଅଚଳ ଦଶମିକ id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K/4; j += 32) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା m = 0; m < 32; ++m) {
            uint8_t q = 0;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା xi = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
