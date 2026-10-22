#pragma once

#অন্তর্ভাব "common.h"

শূন্য quantize_q1_0(device ধ্রুবক দশমিক * src, device block_q1_0 & dst) {
    দশমিক যোগ_abs = 0.0f;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK1_0; j++) {
        যোগ_abs += fabs(src[j]);
    }
    dst.d = যোগ_abs / QK1_0;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    জন্য (পূর্ণসংখ্যা j = 0; j < QK1_0; j++) {
        যদি (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

শূন্য quantize_q2_0(device ধ্রুবক দশমিক * src, device block_q2_0 & dst) {
    দশমিক amax = 0.0f;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK2_0; j++) {
        দশমিক a = fabs(src[j]);
        যদি (a > amax) amax = a;
    }
    ধ্রুবক দশমিক d = amax;
    dst.d = d;

    ধ্রুবক দশমিক id = d > 0.0f ? 1.0f / d : 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    জন্য (পূর্ণসংখ্যা j = 0; j < QK2_0; j++) {
        পূর্ণসংখ্যা q = (পূর্ণসংখ্যা)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

শূন্য quantize_q4_0(device ধ্রুবক দশমিক * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    দশমিক amax = 0.0f; // absolute max
    দশমিক max  = 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_0; j++) {
        ধ্রুবক দশমিক v = src[j];
        যদি (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ধ্রুবক দশমিক d = max / -8;
    ধ্রুবক দশমিক id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_0/2; ++j) {
        ধ্রুবক দশমিক x0 = src[0       + j]*id;
        ধ্রুবক দশমিক x1 = src[QK4_0/2 + j]*id;

        ধ্রুবক uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ধ্রুবক uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

শূন্য quantize_q4_1(device ধ্রুবক দশমিক * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    দশমিক min = FLT_MAX;
    দশমিক max = -FLT_MAX;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_1; j++) {
        ধ্রুবক দশমিক v = src[j];
        যদি (min > v) min = v;
        যদি (max < v) max = v;
    }

    ধ্রুবক দশমিক d = (max - min) / ((1 << 4) - 1);
    ধ্রুবক দশমিক id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_1/2; ++j) {
        ধ্রুবক দশমিক x0 = (src[0       + j] - min)*id;
        ধ্রুবক দশমিক x1 = (src[QK4_1/2 + j] - min)*id;

        ধ্রুবক uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ধ্রুবক uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

শূন্য quantize_q5_0(device ধ্রুবক দশমিক * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    দশমিক amax = 0.0f; // absolute max
    দশমিক max  = 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK5_0; j++) {
        ধ্রুবক দশমিক v = src[j];
        যদি (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ধ্রুবক দশমিক d = max / -16;
    ধ্রুবক দশমিক id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK5_0/2; ++j) {
        ধ্রুবক দশমিক x0 = src[0       + j]*id;
        ধ্রুবক দশমিক x1 = src[QK5_0/2 + j]*id;

        ধ্রুবক uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ধ্রুবক uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ধ্রুবক uint8_t * qh8 = (thread ধ্রুবক uint8_t *)&qh;

    জন্য (পূর্ণসংখ্যা j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

শূন্য quantize_q5_1(device ধ্রুবক দশমিক * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    দশমিক max = src[0];
    দশমিক min = src[0];

    জন্য (পূর্ণসংখ্যা j = 1; j < QK5_1; j++) {
        ধ্রুবক দশমিক v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ধ্রুবক দশমিক d = (max - min) / 31;
    ধ্রুবক দশমিক id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK5_1/2; ++j) {
        ধ্রুবক দশমিক x0 = (src[0       + j] - min)*id;
        ধ্রুবক দশমিক x1 = (src[QK5_1/2 + j] - min)*id;

        ধ্রুবক uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ধ্রুবক uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ধ্রুবক uint8_t * qh8 = (thread ধ্রুবক uint8_t *)&qh;

    জন্য (পূর্ণসংখ্যা j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

শূন্য quantize_q8_0(device ধ্রুবক দশমিক * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    দশমিক amax = 0.0f; // absolute max

    জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; j++) {
        ধ্রুবক দশমিক v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ধ্রুবক দশমিক d = amax / ((1 << 7) - 1);
    ধ্রুবক দশমিক id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; ++j) {
        ধ্রুবক দশমিক x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

শূন্য quantize_iq4_nl(device ধ্রুবক দশমিক * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    দশমিক amax = 0.0f; // absolute max
    দশমিক max  = 0.0f;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_NL; j++) {
        ধ্রুবক দশমিক v = src[j];
        যদি (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ধ্রুবক দশমিক d = max / kvalues_iq4nl_f[0];
    ধ্রুবক দশমিক id = d ? 1.0f/d : 0.0f;

    দশমিক sumqx = 0, sumq2 = 0;
    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_NL/2; ++j) {
        ধ্রুবক দশমিক x0 = src[0        + j]*id;
        ধ্রুবক দশমিক x1 = src[QK4_NL/2 + j]*id;

        ধ্রুবক uint8_t xi0 = best_সূচক_int8(16, kvalues_iq4nl_f, x0);
        ধ্রুবক uint8_t xi1 = best_সূচক_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ধ্রুবক দশমিক v0 = kvalues_iq4nl_f[xi0];
        ধ্রুবক দশমিক v1 = kvalues_iq4nl_f[xi1];
        ধ্রুবক দশমিক w0 = src[0        + j]*src[0        + j];
        ধ্রুবক দশমিক w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

শূন্য quantize_tq2_0(device ধ্রুবক দশমিক * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    দশমিক amax = 0.0f; // absolute max

    জন্য (পূর্ণসংখ্যা j = 0; j < QK_K; j++) {
        ধ্রুবক দশমিক v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ধ্রুবক দশমিক d = amax;
    ধ্রুবক দশমিক id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK_K/4; j += 32) {
        জন্য (পূর্ণসংখ্যা m = 0; m < 32; ++m) {
            uint8_t q = 0;
            জন্য (পূর্ণসংখ্যা n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                পূর্ণসংখ্যা xi = (পূর্ণসংখ্যা)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
