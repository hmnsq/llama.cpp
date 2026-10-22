#pragma once

#અન્તર્ભાવ "common.h"

શૂન્ય quantize_q1_0(device અચલ દશાંશ * src, device block_q1_0 & dst) {
    દશાંશ sum_abs = 0.0f;
    માટે (પૂર્ણાંક j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    માટે (પૂર્ણાંક j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    માટે (પૂર્ણાંક j = 0; j < QK1_0; j++) {
        જો (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

શૂન્ય quantize_q2_0(device અચલ દશાંશ * src, device block_q2_0 & dst) {
    દશાંશ amax = 0.0f;
    માટે (પૂર્ણાંક j = 0; j < QK2_0; j++) {
        દશાંશ a = fabs(src[j]);
        જો (a > amax) amax = a;
    }
    અચલ દશાંશ d = amax;
    dst.d = d;

    અચલ દશાંશ id = d > 0.0f ? 1.0f / d : 0.0f;

    માટે (પૂર્ણાંક j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    માટે (પૂર્ણાંક j = 0; j < QK2_0; j++) {
        પૂર્ણાંક q = (પૂર્ણાંક)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

શૂન્ય quantize_q4_0(device અચલ દશાંશ * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    દશાંશ amax = 0.0f; // absolute max
    દશાંશ max  = 0.0f;

    માટે (પૂર્ણાંક j = 0; j < QK4_0; j++) {
        અચલ દશાંશ v = src[j];
        જો (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    અચલ દશાંશ d = max / -8;
    અચલ દશાંશ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    માટે (પૂર્ણાંક j = 0; j < QK4_0/2; ++j) {
        અચલ દશાંશ x0 = src[0       + j]*id;
        અચલ દશાંશ x1 = src[QK4_0/2 + j]*id;

        અચલ uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        અચલ uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

શૂન્ય quantize_q4_1(device અચલ દશાંશ * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    દશાંશ min = FLT_MAX;
    દશાંશ max = -FLT_MAX;

    માટે (પૂર્ણાંક j = 0; j < QK4_1; j++) {
        અચલ દશાંશ v = src[j];
        જો (min > v) min = v;
        જો (max < v) max = v;
    }

    અચલ દશાંશ d = (max - min) / ((1 << 4) - 1);
    અચલ દશાંશ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    માટે (પૂર્ણાંક j = 0; j < QK4_1/2; ++j) {
        અચલ દશાંશ x0 = (src[0       + j] - min)*id;
        અચલ દશાંશ x1 = (src[QK4_1/2 + j] - min)*id;

        અચલ uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        અચલ uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

શૂન્ય quantize_q5_0(device અચલ દશાંશ * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    દશાંશ amax = 0.0f; // absolute max
    દશાંશ max  = 0.0f;

    માટે (પૂર્ણાંક j = 0; j < QK5_0; j++) {
        અચલ દશાંશ v = src[j];
        જો (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    અચલ દશાંશ d = max / -16;
    અચલ દશાંશ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    માટે (પૂર્ણાંક j = 0; j < QK5_0/2; ++j) {
        અચલ દશાંશ x0 = src[0       + j]*id;
        અચલ દશાંશ x1 = src[QK5_0/2 + j]*id;

        અચલ uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        અચલ uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread અચલ uint8_t * qh8 = (thread અચલ uint8_t *)&qh;

    માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

શૂન્ય quantize_q5_1(device અચલ દશાંશ * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    દશાંશ max = src[0];
    દશાંશ min = src[0];

    માટે (પૂર્ણાંક j = 1; j < QK5_1; j++) {
        અચલ દશાંશ v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    અચલ દશાંશ d = (max - min) / 31;
    અચલ દશાંશ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    માટે (પૂર્ણાંક j = 0; j < QK5_1/2; ++j) {
        અચલ દશાંશ x0 = (src[0       + j] - min)*id;
        અચલ દશાંશ x1 = (src[QK5_1/2 + j] - min)*id;

        અચલ uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        અચલ uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread અચલ uint8_t * qh8 = (thread અચલ uint8_t *)&qh;

    માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

શૂન્ય quantize_q8_0(device અચલ દશાંશ * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    દશાંશ amax = 0.0f; // absolute max

    માટે (પૂર્ણાંક j = 0; j < QK8_0; j++) {
        અચલ દશાંશ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    અચલ દશાંશ d = amax / ((1 << 7) - 1);
    અચલ દશાંશ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    માટે (પૂર્ણાંક j = 0; j < QK8_0; ++j) {
        અચલ દશાંશ x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

શૂન્ય quantize_iq4_nl(device અચલ દશાંશ * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    દશાંશ amax = 0.0f; // absolute max
    દશાંશ max  = 0.0f;

    માટે (પૂર્ણાંક j = 0; j < QK4_NL; j++) {
        અચલ દશાંશ v = src[j];
        જો (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    અચલ દશાંશ d = max / kvalues_iq4nl_f[0];
    અચલ દશાંશ id = d ? 1.0f/d : 0.0f;

    દશાંશ sumqx = 0, sumq2 = 0;
    માટે (પૂર્ણાંક j = 0; j < QK4_NL/2; ++j) {
        અચલ દશાંશ x0 = src[0        + j]*id;
        અચલ દશાંશ x1 = src[QK4_NL/2 + j]*id;

        અચલ uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        અચલ uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        અચલ દશાંશ v0 = kvalues_iq4nl_f[xi0];
        અચલ દશાંશ v1 = kvalues_iq4nl_f[xi1];
        અચલ દશાંશ w0 = src[0        + j]*src[0        + j];
        અચલ દશાંશ w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

શૂન્ય quantize_tq2_0(device અચલ દશાંશ * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    દશાંશ amax = 0.0f; // absolute max

    માટે (પૂર્ણાંક j = 0; j < QK_K; j++) {
        અચલ દશાંશ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    અચલ દશાંશ d = amax;
    અચલ દશાંશ id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    માટે (પૂર્ણાંક j = 0; j < QK_K/4; j += 32) {
        માટે (પૂર્ણાંક m = 0; m < 32; ++m) {
            uint8_t q = 0;
            માટે (પૂર્ણાંક n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                પૂર્ણાંક xi = (પૂર્ણાંક)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
