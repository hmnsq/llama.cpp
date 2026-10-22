#pragma once

#அந்தர்பாவ "common.h"

வெறுமை quantize_q1_0(device மாறிலி தசம * src, device block_q1_0 & dst) {
    தசம sum_abs = 0.0f;
    க்கு (முழு j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    க்கு (முழு j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    க்கு (முழு j = 0; j < QK1_0; j++) {
        எனில் (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

வெறுமை quantize_q2_0(device மாறிலி தசம * src, device block_q2_0 & dst) {
    தசம amax = 0.0f;
    க்கு (முழு j = 0; j < QK2_0; j++) {
        தசம a = fabs(src[j]);
        எனில் (a > amax) amax = a;
    }
    மாறிலி தசம d = amax;
    dst.d = d;

    மாறிலி தசம id = d > 0.0f ? 1.0f / d : 0.0f;

    க்கு (முழு j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    க்கு (முழு j = 0; j < QK2_0; j++) {
        முழு q = (முழு)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

வெறுமை quantize_q4_0(device மாறிலி தசம * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    தசம amax = 0.0f; // absolute max
    தசம max  = 0.0f;

    க்கு (முழு j = 0; j < QK4_0; j++) {
        மாறிலி தசம v = src[j];
        எனில் (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    மாறிலி தசம d = max / -8;
    மாறிலி தசம id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    க்கு (முழு j = 0; j < QK4_0/2; ++j) {
        மாறிலி தசம x0 = src[0       + j]*id;
        மாறிலி தசம x1 = src[QK4_0/2 + j]*id;

        மாறிலி uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        மாறிலி uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

வெறுமை quantize_q4_1(device மாறிலி தசம * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    தசம min = FLT_MAX;
    தசம max = -FLT_MAX;

    க்கு (முழு j = 0; j < QK4_1; j++) {
        மாறிலி தசம v = src[j];
        எனில் (min > v) min = v;
        எனில் (max < v) max = v;
    }

    மாறிலி தசம d = (max - min) / ((1 << 4) - 1);
    மாறிலி தசம id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    க்கு (முழு j = 0; j < QK4_1/2; ++j) {
        மாறிலி தசம x0 = (src[0       + j] - min)*id;
        மாறிலி தசம x1 = (src[QK4_1/2 + j] - min)*id;

        மாறிலி uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        மாறிலி uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

வெறுமை quantize_q5_0(device மாறிலி தசம * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    தசம amax = 0.0f; // absolute max
    தசம max  = 0.0f;

    க்கு (முழு j = 0; j < QK5_0; j++) {
        மாறிலி தசம v = src[j];
        எனில் (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    மாறிலி தசம d = max / -16;
    மாறிலி தசம id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    க்கு (முழு j = 0; j < QK5_0/2; ++j) {
        மாறிலி தசம x0 = src[0       + j]*id;
        மாறிலி தசம x1 = src[QK5_0/2 + j]*id;

        மாறிலி uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        மாறிலி uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread மாறிலி uint8_t * qh8 = (thread மாறிலி uint8_t *)&qh;

    க்கு (முழு j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

வெறுமை quantize_q5_1(device மாறிலி தசம * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    தசம max = src[0];
    தசம min = src[0];

    க்கு (முழு j = 1; j < QK5_1; j++) {
        மாறிலி தசம v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    மாறிலி தசம d = (max - min) / 31;
    மாறிலி தசம id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    க்கு (முழு j = 0; j < QK5_1/2; ++j) {
        மாறிலி தசம x0 = (src[0       + j] - min)*id;
        மாறிலி தசம x1 = (src[QK5_1/2 + j] - min)*id;

        மாறிலி uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        மாறிலி uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread மாறிலி uint8_t * qh8 = (thread மாறிலி uint8_t *)&qh;

    க்கு (முழு j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

வெறுமை quantize_q8_0(device மாறிலி தசம * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    தசம amax = 0.0f; // absolute max

    க்கு (முழு j = 0; j < QK8_0; j++) {
        மாறிலி தசம v = src[j];
        amax = MAX(amax, fabs(v));
    }

    மாறிலி தசம d = amax / ((1 << 7) - 1);
    மாறிலி தசம id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    க்கு (முழு j = 0; j < QK8_0; ++j) {
        மாறிலி தசம x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

வெறுமை quantize_iq4_nl(device மாறிலி தசம * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    தசம amax = 0.0f; // absolute max
    தசம max  = 0.0f;

    க்கு (முழு j = 0; j < QK4_NL; j++) {
        மாறிலி தசம v = src[j];
        எனில் (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    மாறிலி தசம d = max / kvalues_iq4nl_f[0];
    மாறிலி தசம id = d ? 1.0f/d : 0.0f;

    தசம sumqx = 0, sumq2 = 0;
    க்கு (முழு j = 0; j < QK4_NL/2; ++j) {
        மாறிலி தசம x0 = src[0        + j]*id;
        மாறிலி தசம x1 = src[QK4_NL/2 + j]*id;

        மாறிலி uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        மாறிலி uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        மாறிலி தசம v0 = kvalues_iq4nl_f[xi0];
        மாறிலி தசம v1 = kvalues_iq4nl_f[xi1];
        மாறிலி தசம w0 = src[0        + j]*src[0        + j];
        மாறிலி தசம w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

வெறுமை quantize_tq2_0(device மாறிலி தசம * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    தசம amax = 0.0f; // absolute max

    க்கு (முழு j = 0; j < QK_K; j++) {
        மாறிலி தசம v = src[j];
        amax = MAX(amax, fabs(v));
    }

    மாறிலி தசம d = amax;
    மாறிலி தசம id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    க்கு (முழு j = 0; j < QK_K/4; j += 32) {
        க்கு (முழு m = 0; m < 32; ++m) {
            uint8_t q = 0;
            க்கு (முழு n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                முழு xi = (முழு)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
