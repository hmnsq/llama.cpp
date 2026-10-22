#निर्देशः once

#अन्तर्भाव "common.h"

शून्य quantize_q1_0(device ध्रुव भिन्न * src, device खण्डम्_q1_0 & dst) {
    भिन्न योग_abs = 0.0f;
    प्रति (पूर्णाङ्क j = 0; j < QK1_0; j++) {
        योग_abs += fabs(src[j]);
    }
    dst.d = योग_abs / QK1_0;

    प्रति (पूर्णाङ्क j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    प्रति (पूर्णाङ्क j = 0; j < QK1_0; j++) {
        यदि (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

शून्य quantize_q2_0(device ध्रुव भिन्न * src, device खण्डम्_q2_0 & dst) {
    भिन्न amax = 0.0f;
    प्रति (पूर्णाङ्क j = 0; j < QK2_0; j++) {
        भिन्न a = fabs(src[j]);
        यदि (a > amax) amax = a;
    }
    ध्रुव भिन्न d = amax;
    dst.d = d;

    ध्रुव भिन्न id = d > 0.0f ? 1.0f / d : 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    प्रति (पूर्णाङ्क j = 0; j < QK2_0; j++) {
        पूर्णाङ्क q = (पूर्णाङ्क)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

शून्य quantize_q4_0(device ध्रुव भिन्न * src, device खण्डम्_q4_0 & dst) {
#निर्देशः METAL fp math_बहुलकम्(safe)
    भिन्न amax = 0.0f; // absolute max
    भिन्न max  = 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK4_0; j++) {
        ध्रुव भिन्न v = src[j];
        यदि (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ध्रुव भिन्न d = max / -8;
    ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    प्रति (पूर्णाङ्क j = 0; j < QK4_0/2; ++j) {
        ध्रुव भिन्न x0 = src[0       + j]*id;
        ध्रुव भिन्न x1 = src[QK4_0/2 + j]*id;

        ध्रुव uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ध्रुव uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

शून्य quantize_q4_1(device ध्रुव भिन्न * src, device खण्डम्_q4_1 & dst) {
#निर्देशः METAL fp math_बहुलकम्(safe)
    भिन्न min = FLT_MAX;
    भिन्न max = -FLT_MAX;

    प्रति (पूर्णाङ्क j = 0; j < QK4_1; j++) {
        ध्रुव भिन्न v = src[j];
        यदि (min > v) min = v;
        यदि (max < v) max = v;
    }

    ध्रुव भिन्न d = (max - min) / ((1 << 4) - 1);
    ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    प्रति (पूर्णाङ्क j = 0; j < QK4_1/2; ++j) {
        ध्रुव भिन्न x0 = (src[0       + j] - min)*id;
        ध्रुव भिन्न x1 = (src[QK4_1/2 + j] - min)*id;

        ध्रुव uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ध्रुव uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

शून्य quantize_q5_0(device ध्रुव भिन्न * src, device खण्डम्_q5_0 & dst) {
#निर्देशः METAL fp math_बहुलकम्(safe)
    भिन्न amax = 0.0f; // absolute max
    भिन्न max  = 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK5_0; j++) {
        ध्रुव भिन्न v = src[j];
        यदि (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ध्रुव भिन्न d = max / -16;
    ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    प्रति (पूर्णाङ्क j = 0; j < QK5_0/2; ++j) {
        ध्रुव भिन्न x0 = src[0       + j]*id;
        ध्रुव भिन्न x1 = src[QK5_0/2 + j]*id;

        ध्रुव uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ध्रुव uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ध्रुव uint8_t * qh8 = (thread ध्रुव uint8_t *)&qh;

    प्रति (पूर्णाङ्क j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

शून्य quantize_q5_1(device ध्रुव भिन्न * src, device खण्डम्_q5_1 & dst) {
#निर्देशः METAL fp math_बहुलकम्(safe)
    भिन्न max = src[0];
    भिन्न min = src[0];

    प्रति (पूर्णाङ्क j = 1; j < QK5_1; j++) {
        ध्रुव भिन्न v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ध्रुव भिन्न d = (max - min) / 31;
    ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    प्रति (पूर्णाङ्क j = 0; j < QK5_1/2; ++j) {
        ध्रुव भिन्न x0 = (src[0       + j] - min)*id;
        ध्रुव भिन्न x1 = (src[QK5_1/2 + j] - min)*id;

        ध्रुव uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ध्रुव uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ध्रुव uint8_t * qh8 = (thread ध्रुव uint8_t *)&qh;

    प्रति (पूर्णाङ्क j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

शून्य quantize_q8_0(device ध्रुव भिन्न * src, device खण्डम्_q8_0 & dst) {
#निर्देशः METAL fp math_बहुलकम्(safe)
    भिन्न amax = 0.0f; // absolute max

    प्रति (पूर्णाङ्क j = 0; j < QK8_0; j++) {
        ध्रुव भिन्न v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ध्रुव भिन्न d = amax / ((1 << 7) - 1);
    ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    प्रति (पूर्णाङ्क j = 0; j < QK8_0; ++j) {
        ध्रुव भिन्न x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

शून्य quantize_iq4_nl(device ध्रुव भिन्न * src, device खण्डम्_iq4_nl & dst) {
#निर्देशः METAL fp math_बहुलकम्(safe)
    भिन्न amax = 0.0f; // absolute max
    भिन्न max  = 0.0f;

    प्रति (पूर्णाङ्क j = 0; j < QK4_NL; j++) {
        ध्रुव भिन्न v = src[j];
        यदि (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ध्रुव भिन्न d = max / kvalues_iq4nl_f[0];
    ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;

    भिन्न sumqx = 0, sumq2 = 0;
    प्रति (पूर्णाङ्क j = 0; j < QK4_NL/2; ++j) {
        ध्रुव भिन्न x0 = src[0        + j]*id;
        ध्रुव भिन्न x1 = src[QK4_NL/2 + j]*id;

        ध्रुव uint8_t xi0 = best_सूचकांक_int8(16, kvalues_iq4nl_f, x0);
        ध्रुव uint8_t xi1 = best_सूचकांक_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ध्रुव भिन्न v0 = kvalues_iq4nl_f[xi0];
        ध्रुव भिन्न v1 = kvalues_iq4nl_f[xi1];
        ध्रुव भिन्न w0 = src[0        + j]*src[0        + j];
        ध्रुव भिन्न w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

शून्य quantize_tq2_0(device ध्रुव भिन्न * src, device खण्डम्_tq2_0 & dst) {
#निर्देशः METAL fp math_बहुलकम्(safe)
    भिन्न amax = 0.0f; // absolute max

    प्रति (पूर्णाङ्क j = 0; j < QK_K; j++) {
        ध्रुव भिन्न v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ध्रुव भिन्न d = amax;
    ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    प्रति (पूर्णाङ्क j = 0; j < QK_K/4; j += 32) {
        प्रति (पूर्णाङ्क m = 0; m < 32; ++m) {
            uint8_t q = 0;
            प्रति (पूर्णाङ्क n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                पूर्णाङ्क xi = (पूर्णाङ्क)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
