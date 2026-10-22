#pragma once

#ضمن "common.h"

فراغ quantize_q1_0(device ثابت عائم * src, device block_q1_0 & dst) {
    عائم sum_abs = 0.0f;
    עבור (صحيح j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    עבור (صحيح j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    עבור (صحيح j = 0; j < QK1_0; j++) {
        אם (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

فراغ quantize_q2_0(device ثابت عائم * src, device block_q2_0 & dst) {
    عائم amax = 0.0f;
    עבור (صحيح j = 0; j < QK2_0; j++) {
        عائم a = fabs(src[j]);
        אם (a > amax) amax = a;
    }
    ثابت عائم d = amax;
    dst.d = d;

    ثابت عائم id = d > 0.0f ? 1.0f / d : 0.0f;

    עבור (صحيح j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    עבור (صحيح j = 0; j < QK2_0; j++) {
        صحيح q = (صحيح)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

فراغ quantize_q4_0(device ثابت عائم * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    عائم amax = 0.0f; // absolute max
    عائم max  = 0.0f;

    עבור (صحيح j = 0; j < QK4_0; j++) {
        ثابت عائم v = src[j];
        אם (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ثابت عائم d = max / -8;
    ثابت عائم id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    עבור (صحيح j = 0; j < QK4_0/2; ++j) {
        ثابت عائم x0 = src[0       + j]*id;
        ثابت عائم x1 = src[QK4_0/2 + j]*id;

        ثابت uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ثابت uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

فراغ quantize_q4_1(device ثابت عائم * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    عائم min = FLT_MAX;
    عائم max = -FLT_MAX;

    עבור (صحيح j = 0; j < QK4_1; j++) {
        ثابت عائم v = src[j];
        אם (min > v) min = v;
        אם (max < v) max = v;
    }

    ثابت عائم d = (max - min) / ((1 << 4) - 1);
    ثابت عائم id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    עבור (صحيح j = 0; j < QK4_1/2; ++j) {
        ثابت عائم x0 = (src[0       + j] - min)*id;
        ثابت عائم x1 = (src[QK4_1/2 + j] - min)*id;

        ثابت uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ثابت uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

فراغ quantize_q5_0(device ثابت عائم * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    عائم amax = 0.0f; // absolute max
    عائم max  = 0.0f;

    עבור (صحيح j = 0; j < QK5_0; j++) {
        ثابت عائم v = src[j];
        אם (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ثابت عائم d = max / -16;
    ثابت عائم id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    עבור (صحيح j = 0; j < QK5_0/2; ++j) {
        ثابت عائم x0 = src[0       + j]*id;
        ثابت عائم x1 = src[QK5_0/2 + j]*id;

        ثابت uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ثابت uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ثابت uint8_t * qh8 = (thread ثابت uint8_t *)&qh;

    עבור (صحيح j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

فراغ quantize_q5_1(device ثابت عائم * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    عائم max = src[0];
    عائم min = src[0];

    עבור (صحيح j = 1; j < QK5_1; j++) {
        ثابت عائم v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ثابت عائم d = (max - min) / 31;
    ثابت عائم id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    עבור (صحيح j = 0; j < QK5_1/2; ++j) {
        ثابت عائم x0 = (src[0       + j] - min)*id;
        ثابت عائم x1 = (src[QK5_1/2 + j] - min)*id;

        ثابت uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ثابت uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ثابت uint8_t * qh8 = (thread ثابت uint8_t *)&qh;

    עבור (صحيح j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

فراغ quantize_q8_0(device ثابت عائم * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    عائم amax = 0.0f; // absolute max

    עבור (صحيح j = 0; j < QK8_0; j++) {
        ثابت عائم v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ثابت عائم d = amax / ((1 << 7) - 1);
    ثابت عائم id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    עבור (صحيح j = 0; j < QK8_0; ++j) {
        ثابت عائم x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

فراغ quantize_iq4_nl(device ثابت عائم * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    عائم amax = 0.0f; // absolute max
    عائم max  = 0.0f;

    עבור (صحيح j = 0; j < QK4_NL; j++) {
        ثابت عائم v = src[j];
        אם (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ثابت عائم d = max / kvalues_iq4nl_f[0];
    ثابت عائم id = d ? 1.0f/d : 0.0f;

    عائم sumqx = 0, sumq2 = 0;
    עבור (صحيح j = 0; j < QK4_NL/2; ++j) {
        ثابت عائم x0 = src[0        + j]*id;
        ثابت عائم x1 = src[QK4_NL/2 + j]*id;

        ثابت uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        ثابت uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ثابت عائم v0 = kvalues_iq4nl_f[xi0];
        ثابت عائم v1 = kvalues_iq4nl_f[xi1];
        ثابت عائم w0 = src[0        + j]*src[0        + j];
        ثابت عائم w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

فراغ quantize_tq2_0(device ثابت عائم * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    عائم amax = 0.0f; // absolute max

    עבור (صحيح j = 0; j < QK_K; j++) {
        ثابت عائم v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ثابت عائم d = amax;
    ثابت عائم id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    עבור (صحيح j = 0; j < QK_K/4; j += 32) {
        עבור (صحيح m = 0; m < 32; ++m) {
            uint8_t q = 0;
            עבור (صحيح n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                صحيح xi = (صحيح)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
