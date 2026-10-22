#आशय once

#समावेश "common.h"

व्योम quantize_q1_0(device स्थिर भग्न * src, device block_q1_0 & dst) {
    भग्न योग_abs = 0.0f;
    क्रम (पूर्णांक j = 0; j < QK1_0; j++) {
        योग_abs += भ_असल(src[j]);
    }
    dst.d = योग_abs / QK1_0;

    क्रम (पूर्णांक j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    क्रम (पूर्णांक j = 0; j < QK1_0; j++) {
        अगर (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

व्योम quantize_q2_0(device स्थिर भग्न * src, device block_q2_0 & dst) {
    भग्न amax = 0.0f;
    क्रम (पूर्णांक j = 0; j < QK2_0; j++) {
        भग्न a = भ_असल(src[j]);
        अगर (a > amax) amax = a;
    }
    स्थिर भग्न d = amax;
    dst.d = d;

    स्थिर भग्न id = d > 0.0f ? 1.0f / d : 0.0f;

    क्रम (पूर्णांक j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    क्रम (पूर्णांक j = 0; j < QK2_0; j++) {
        पूर्णांक q = (पूर्णांक)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

व्योम quantize_q4_0(device स्थिर भग्न * src, device block_q4_0 & dst) {
#आशय METAL fp math_mode(safe)
    भग्न amax = 0.0f; // absolute max
    भग्न max  = 0.0f;

    क्रम (पूर्णांक j = 0; j < QK4_0; j++) {
        स्थिर भग्न v = src[j];
        अगर (amax < भ_असल(v)) {
            amax = भ_असल(v);
            max  = v;
        }
    }

    स्थिर भग्न d = max / -8;
    स्थिर भग्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    क्रम (पूर्णांक j = 0; j < QK4_0/2; ++j) {
        स्थिर भग्न x0 = src[0       + j]*id;
        स्थिर भग्न x1 = src[QK4_0/2 + j]*id;

        स्थिर uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        स्थिर uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

व्योम quantize_q4_1(device स्थिर भग्न * src, device block_q4_1 & dst) {
#आशय METAL fp math_mode(safe)
    भग्न min = भग्न_उच्च;
    भग्न max = -भग्न_उच्च;

    क्रम (पूर्णांक j = 0; j < QK4_1; j++) {
        स्थिर भग्न v = src[j];
        अगर (min > v) min = v;
        अगर (max < v) max = v;
    }

    स्थिर भग्न d = (max - min) / ((1 << 4) - 1);
    स्थिर भग्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    क्रम (पूर्णांक j = 0; j < QK4_1/2; ++j) {
        स्थिर भग्न x0 = (src[0       + j] - min)*id;
        स्थिर भग्न x1 = (src[QK4_1/2 + j] - min)*id;

        स्थिर uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        स्थिर uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

व्योम quantize_q5_0(device स्थिर भग्न * src, device block_q5_0 & dst) {
#आशय METAL fp math_mode(safe)
    भग्न amax = 0.0f; // absolute max
    भग्न max  = 0.0f;

    क्रम (पूर्णांक j = 0; j < QK5_0; j++) {
        स्थिर भग्न v = src[j];
        अगर (amax < भ_असल(v)) {
            amax = भ_असल(v);
            max  = v;
        }
    }

    स्थिर भग्न d = max / -16;
    स्थिर भग्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    क्रम (पूर्णांक j = 0; j < QK5_0/2; ++j) {
        स्थिर भग्न x0 = src[0       + j]*id;
        स्थिर भग्न x1 = src[QK5_0/2 + j]*id;

        स्थिर uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        स्थिर uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread स्थिर uint8_t * qh8 = (thread स्थिर uint8_t *)&qh;

    क्रम (पूर्णांक j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

व्योम quantize_q5_1(device स्थिर भग्न * src, device block_q5_1 & dst) {
#आशय METAL fp math_mode(safe)
    भग्न max = src[0];
    भग्न min = src[0];

    क्रम (पूर्णांक j = 1; j < QK5_1; j++) {
        स्थिर भग्न v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    स्थिर भग्न d = (max - min) / 31;
    स्थिर भग्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    क्रम (पूर्णांक j = 0; j < QK5_1/2; ++j) {
        स्थिर भग्न x0 = (src[0       + j] - min)*id;
        स्थिर भग्न x1 = (src[QK5_1/2 + j] - min)*id;

        स्थिर uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        स्थिर uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread स्थिर uint8_t * qh8 = (thread स्थिर uint8_t *)&qh;

    क्रम (पूर्णांक j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

व्योम quantize_q8_0(device स्थिर भग्न * src, device block_q8_0 & dst) {
#आशय METAL fp math_mode(safe)
    भग्न amax = 0.0f; // absolute max

    क्रम (पूर्णांक j = 0; j < QK8_0; j++) {
        स्थिर भग्न v = src[j];
        amax = MAX(amax, भ_असल(v));
    }

    स्थिर भग्न d = amax / ((1 << 7) - 1);
    स्थिर भग्न id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    क्रम (पूर्णांक j = 0; j < QK8_0; ++j) {
        स्थिर भग्न x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

व्योम quantize_iq4_nl(device स्थिर भग्न * src, device block_iq4_nl & dst) {
#आशय METAL fp math_mode(safe)
    भग्न amax = 0.0f; // absolute max
    भग्न max  = 0.0f;

    क्रम (पूर्णांक j = 0; j < QK4_NL; j++) {
        स्थिर भग्न v = src[j];
        अगर (amax < भ_असल(v)) {
            amax = भ_असल(v);
            max  = v;
        }
    }

    स्थिर भग्न d = max / kvalues_iq4nl_f[0];
    स्थिर भग्न id = d ? 1.0f/d : 0.0f;

    भग्न sumqx = 0, sumq2 = 0;
    क्रम (पूर्णांक j = 0; j < QK4_NL/2; ++j) {
        स्थिर भग्न x0 = src[0        + j]*id;
        स्थिर भग्न x1 = src[QK4_NL/2 + j]*id;

        स्थिर uint8_t xi0 = best_सूचकांक_int8(16, kvalues_iq4nl_f, x0);
        स्थिर uint8_t xi1 = best_सूचकांक_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        स्थिर भग्न v0 = kvalues_iq4nl_f[xi0];
        स्थिर भग्न v1 = kvalues_iq4nl_f[xi1];
        स्थिर भग्न w0 = src[0        + j]*src[0        + j];
        स्थिर भग्न w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

व्योम quantize_tq2_0(device स्थिर भग्न * src, device block_tq2_0 & dst) {
#आशय METAL fp math_mode(safe)
    भग्न amax = 0.0f; // absolute max

    क्रम (पूर्णांक j = 0; j < QK_K; j++) {
        स्थिर भग्न v = src[j];
        amax = MAX(amax, भ_असल(v));
    }

    स्थिर भग्न d = amax;
    स्थिर भग्न id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    क्रम (पूर्णांक j = 0; j < QK_K/4; j += 32) {
        क्रम (पूर्णांक m = 0; m < 32; ++m) {
            uint8_t q = 0;
            क्रम (पूर्णांक n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                पूर्णांक xi = (पूर्णांक)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
