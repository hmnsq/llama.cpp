#pragma once

#অন্তর্ভাৱ "common.h"

শূন্য quantize_q1_0(device ধ্রুৱ ভিন্ন * src, device block_q1_0 & dst) {
    ভিন্ন sum_abs = 0.0f;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    বাবে (পূর্ণাঙ্ক j = 0; j < QK1_0; j++) {
        যদি (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

শূন্য quantize_q2_0(device ধ্রুৱ ভিন্ন * src, device block_q2_0 & dst) {
    ভিন্ন amax = 0.0f;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK2_0; j++) {
        ভিন্ন a = fabs(src[j]);
        যদি (a > amax) amax = a;
    }
    ধ্রুৱ ভিন্ন d = amax;
    dst.d = d;

    ধ্রুৱ ভিন্ন id = d > 0.0f ? 1.0f / d : 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    বাবে (পূর্ণাঙ্ক j = 0; j < QK2_0; j++) {
        পূর্ণাঙ্ক q = (পূর্ণাঙ্ক)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

শূন্য quantize_q4_0(device ধ্রুৱ ভিন্ন * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    ভিন্ন amax = 0.0f; // absolute max
    ভিন্ন max  = 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_0; j++) {
        ধ্রুৱ ভিন্ন v = src[j];
        যদি (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ধ্রুৱ ভিন্ন d = max / -8;
    ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_0/2; ++j) {
        ধ্রুৱ ভিন্ন x0 = src[0       + j]*id;
        ধ্রুৱ ভিন্ন x1 = src[QK4_0/2 + j]*id;

        ধ্রুৱ uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ধ্রুৱ uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

শূন্য quantize_q4_1(device ধ্রুৱ ভিন্ন * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    ভিন্ন min = FLT_MAX;
    ভিন্ন max = -FLT_MAX;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_1; j++) {
        ধ্রুৱ ভিন্ন v = src[j];
        যদি (min > v) min = v;
        যদি (max < v) max = v;
    }

    ধ্রুৱ ভিন্ন d = (max - min) / ((1 << 4) - 1);
    ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_1/2; ++j) {
        ধ্রুৱ ভিন্ন x0 = (src[0       + j] - min)*id;
        ধ্রুৱ ভিন্ন x1 = (src[QK4_1/2 + j] - min)*id;

        ধ্রুৱ uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ধ্রুৱ uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

শূন্য quantize_q5_0(device ধ্রুৱ ভিন্ন * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    ভিন্ন amax = 0.0f; // absolute max
    ভিন্ন max  = 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK5_0; j++) {
        ধ্রুৱ ভিন্ন v = src[j];
        যদি (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ধ্রুৱ ভিন্ন d = max / -16;
    ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK5_0/2; ++j) {
        ধ্রুৱ ভিন্ন x0 = src[0       + j]*id;
        ধ্রুৱ ভিন্ন x1 = src[QK5_0/2 + j]*id;

        ধ্রুৱ uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ধ্রুৱ uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ধ্রুৱ uint8_t * qh8 = (thread ধ্রুৱ uint8_t *)&qh;

    বাবে (পূর্ণাঙ্ক j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

শূন্য quantize_q5_1(device ধ্রুৱ ভিন্ন * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    ভিন্ন max = src[0];
    ভিন্ন min = src[0];

    বাবে (পূর্ণাঙ্ক j = 1; j < QK5_1; j++) {
        ধ্রুৱ ভিন্ন v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ধ্রুৱ ভিন্ন d = (max - min) / 31;
    ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK5_1/2; ++j) {
        ধ্রুৱ ভিন্ন x0 = (src[0       + j] - min)*id;
        ধ্রুৱ ভিন্ন x1 = (src[QK5_1/2 + j] - min)*id;

        ধ্রুৱ uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ধ্রুৱ uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ধ্রুৱ uint8_t * qh8 = (thread ধ্রুৱ uint8_t *)&qh;

    বাবে (পূর্ণাঙ্ক j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

শূন্য quantize_q8_0(device ধ্রুৱ ভিন্ন * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    ভিন্ন amax = 0.0f; // absolute max

    বাবে (পূর্ণাঙ্ক j = 0; j < QK8_0; j++) {
        ধ্রুৱ ভিন্ন v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ধ্রুৱ ভিন্ন d = amax / ((1 << 7) - 1);
    ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK8_0; ++j) {
        ধ্রুৱ ভিন্ন x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

শূন্য quantize_iq4_nl(device ধ্রুৱ ভিন্ন * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    ভিন্ন amax = 0.0f; // absolute max
    ভিন্ন max  = 0.0f;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_NL; j++) {
        ধ্রুৱ ভিন্ন v = src[j];
        যদি (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ধ্রুৱ ভিন্ন d = max / kvalues_iq4nl_f[0];
    ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

    ভিন্ন sumqx = 0, sumq2 = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < QK4_NL/2; ++j) {
        ধ্রুৱ ভিন্ন x0 = src[0        + j]*id;
        ধ্রুৱ ভিন্ন x1 = src[QK4_NL/2 + j]*id;

        ধ্রুৱ uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        ধ্রুৱ uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ধ্রুৱ ভিন্ন v0 = kvalues_iq4nl_f[xi0];
        ধ্রুৱ ভিন্ন v1 = kvalues_iq4nl_f[xi1];
        ধ্রুৱ ভিন্ন w0 = src[0        + j]*src[0        + j];
        ধ্রুৱ ভিন্ন w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

শূন্য quantize_tq2_0(device ধ্রুৱ ভিন্ন * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    ভিন্ন amax = 0.0f; // absolute max

    বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j++) {
        ধ্রুৱ ভিন্ন v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ধ্রুৱ ভিন্ন d = amax;
    ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/4; j += 32) {
        বাবে (পূর্ণাঙ্ক m = 0; m < 32; ++m) {
            uint8_t q = 0;
            বাবে (পূর্ণাঙ্ক n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                পূর্ণাঙ্ক xi = (পূর্ণাঙ্ক)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
