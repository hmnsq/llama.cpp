#pragma once

#شامل "common.h"

تهی quantize_q1_0(device ثابت اعشاری * src, device block_q1_0 & dst) {
    اعشاری sum_abs = 0.0f;
    برای (صحیح j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    برای (صحیح j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    برای (صحیح j = 0; j < QK1_0; j++) {
        اگر (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

تهی quantize_q2_0(device ثابت اعشاری * src, device block_q2_0 & dst) {
    اعشاری amax = 0.0f;
    برای (صحیح j = 0; j < QK2_0; j++) {
        اعشاری a = fabs(src[j]);
        اگر (a > amax) amax = a;
    }
    ثابت اعشاری d = amax;
    dst.d = d;

    ثابت اعشاری id = d > 0.0f ? 1.0f / d : 0.0f;

    برای (صحیح j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    برای (صحیح j = 0; j < QK2_0; j++) {
        صحیح q = (صحیح)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

تهی quantize_q4_0(device ثابت اعشاری * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    اعشاری amax = 0.0f; // absolute max
    اعشاری max  = 0.0f;

    برای (صحیح j = 0; j < QK4_0; j++) {
        ثابت اعشاری v = src[j];
        اگر (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ثابت اعشاری d = max / -8;
    ثابت اعشاری id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    برای (صحیح j = 0; j < QK4_0/2; ++j) {
        ثابت اعشاری x0 = src[0       + j]*id;
        ثابت اعشاری x1 = src[QK4_0/2 + j]*id;

        ثابت uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ثابت uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

تهی quantize_q4_1(device ثابت اعشاری * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    اعشاری min = FLT_MAX;
    اعشاری max = -FLT_MAX;

    برای (صحیح j = 0; j < QK4_1; j++) {
        ثابت اعشاری v = src[j];
        اگر (min > v) min = v;
        اگر (max < v) max = v;
    }

    ثابت اعشاری d = (max - min) / ((1 << 4) - 1);
    ثابت اعشاری id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    برای (صحیح j = 0; j < QK4_1/2; ++j) {
        ثابت اعشاری x0 = (src[0       + j] - min)*id;
        ثابت اعشاری x1 = (src[QK4_1/2 + j] - min)*id;

        ثابت uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ثابت uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

تهی quantize_q5_0(device ثابت اعشاری * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    اعشاری amax = 0.0f; // absolute max
    اعشاری max  = 0.0f;

    برای (صحیح j = 0; j < QK5_0; j++) {
        ثابت اعشاری v = src[j];
        اگر (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ثابت اعشاری d = max / -16;
    ثابت اعشاری id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    برای (صحیح j = 0; j < QK5_0/2; ++j) {
        ثابت اعشاری x0 = src[0       + j]*id;
        ثابت اعشاری x1 = src[QK5_0/2 + j]*id;

        ثابت uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ثابت uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ثابت uint8_t * qh8 = (thread ثابت uint8_t *)&qh;

    برای (صحیح j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

تهی quantize_q5_1(device ثابت اعشاری * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    اعشاری max = src[0];
    اعشاری min = src[0];

    برای (صحیح j = 1; j < QK5_1; j++) {
        ثابت اعشاری v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ثابت اعشاری d = (max - min) / 31;
    ثابت اعشاری id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    برای (صحیح j = 0; j < QK5_1/2; ++j) {
        ثابت اعشاری x0 = (src[0       + j] - min)*id;
        ثابت اعشاری x1 = (src[QK5_1/2 + j] - min)*id;

        ثابت uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ثابت uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ثابت uint8_t * qh8 = (thread ثابت uint8_t *)&qh;

    برای (صحیح j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

تهی quantize_q8_0(device ثابت اعشاری * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    اعشاری amax = 0.0f; // absolute max

    برای (صحیح j = 0; j < QK8_0; j++) {
        ثابت اعشاری v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ثابت اعشاری d = amax / ((1 << 7) - 1);
    ثابت اعشاری id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    برای (صحیح j = 0; j < QK8_0; ++j) {
        ثابت اعشاری x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

تهی quantize_iq4_nl(device ثابت اعشاری * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    اعشاری amax = 0.0f; // absolute max
    اعشاری max  = 0.0f;

    برای (صحیح j = 0; j < QK4_NL; j++) {
        ثابت اعشاری v = src[j];
        اگر (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ثابت اعشاری d = max / kvalues_iq4nl_f[0];
    ثابت اعشاری id = d ? 1.0f/d : 0.0f;

    اعشاری sumqx = 0, sumq2 = 0;
    برای (صحیح j = 0; j < QK4_NL/2; ++j) {
        ثابت اعشاری x0 = src[0        + j]*id;
        ثابت اعشاری x1 = src[QK4_NL/2 + j]*id;

        ثابت uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        ثابت uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ثابت اعشاری v0 = kvalues_iq4nl_f[xi0];
        ثابت اعشاری v1 = kvalues_iq4nl_f[xi1];
        ثابت اعشاری w0 = src[0        + j]*src[0        + j];
        ثابت اعشاری w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

تهی quantize_tq2_0(device ثابت اعشاری * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    اعشاری amax = 0.0f; // absolute max

    برای (صحیح j = 0; j < QK_K; j++) {
        ثابت اعشاری v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ثابت اعشاری d = amax;
    ثابت اعشاری id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    برای (صحیح j = 0; j < QK_K/4; j += 32) {
        برای (صحیح m = 0; m < 32; ++m) {
            uint8_t q = 0;
            برای (صحیح n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                صحیح xi = (صحیح)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
