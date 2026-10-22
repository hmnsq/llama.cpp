#pragma once

#अन्तर्भाव "common.h"

शून्य quantize_q1_0(device स्थिरांक दशांश * src, device block_q1_0 & dst) {
    दशांश sum_abs = 0.0f;
    साठी (पूर्णांक j = 0; j < QK1_0; j++) {
        sum_abs += भ_असल(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    साठी (पूर्णांक j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    साठी (पूर्णांक j = 0; j < QK1_0; j++) {
        जर (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

शून्य quantize_q2_0(device स्थिरांक दशांश * src, device block_q2_0 & dst) {
    दशांश amax = 0.0f;
    साठी (पूर्णांक j = 0; j < QK2_0; j++) {
        दशांश a = भ_असल(src[j]);
        जर (a > amax) amax = a;
    }
    स्थिरांक दशांश d = amax;
    dst.d = d;

    स्थिरांक दशांश id = d > 0.0f ? 1.0f / d : 0.0f;

    साठी (पूर्णांक j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    साठी (पूर्णांक j = 0; j < QK2_0; j++) {
        पूर्णांक q = (पूर्णांक)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

शून्य quantize_q4_0(device स्थिरांक दशांश * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    दशांश amax = 0.0f; // absolute max
    दशांश max  = 0.0f;

    साठी (पूर्णांक j = 0; j < QK4_0; j++) {
        स्थिरांक दशांश v = src[j];
        जर (amax < भ_असल(v)) {
            amax = भ_असल(v);
            max  = v;
        }
    }

    स्थिरांक दशांश d = max / -8;
    स्थिरांक दशांश id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    साठी (पूर्णांक j = 0; j < QK4_0/2; ++j) {
        स्थिरांक दशांश x0 = src[0       + j]*id;
        स्थिरांक दशांश x1 = src[QK4_0/2 + j]*id;

        स्थिरांक uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        स्थिरांक uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

शून्य quantize_q4_1(device स्थिरांक दशांश * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    दशांश min = भग्न_उच्च;
    दशांश max = -भग्न_उच्च;

    साठी (पूर्णांक j = 0; j < QK4_1; j++) {
        स्थिरांक दशांश v = src[j];
        जर (min > v) min = v;
        जर (max < v) max = v;
    }

    स्थिरांक दशांश d = (max - min) / ((1 << 4) - 1);
    स्थिरांक दशांश id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    साठी (पूर्णांक j = 0; j < QK4_1/2; ++j) {
        स्थिरांक दशांश x0 = (src[0       + j] - min)*id;
        स्थिरांक दशांश x1 = (src[QK4_1/2 + j] - min)*id;

        स्थिरांक uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        स्थिरांक uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

शून्य quantize_q5_0(device स्थिरांक दशांश * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    दशांश amax = 0.0f; // absolute max
    दशांश max  = 0.0f;

    साठी (पूर्णांक j = 0; j < QK5_0; j++) {
        स्थिरांक दशांश v = src[j];
        जर (amax < भ_असल(v)) {
            amax = भ_असल(v);
            max  = v;
        }
    }

    स्थिरांक दशांश d = max / -16;
    स्थिरांक दशांश id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    साठी (पूर्णांक j = 0; j < QK5_0/2; ++j) {
        स्थिरांक दशांश x0 = src[0       + j]*id;
        स्थिरांक दशांश x1 = src[QK5_0/2 + j]*id;

        स्थिरांक uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        स्थिरांक uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread स्थिरांक uint8_t * qh8 = (thread स्थिरांक uint8_t *)&qh;

    साठी (पूर्णांक j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

शून्य quantize_q5_1(device स्थिरांक दशांश * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    दशांश max = src[0];
    दशांश min = src[0];

    साठी (पूर्णांक j = 1; j < QK5_1; j++) {
        स्थिरांक दशांश v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    स्थिरांक दशांश d = (max - min) / 31;
    स्थिरांक दशांश id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    साठी (पूर्णांक j = 0; j < QK5_1/2; ++j) {
        स्थिरांक दशांश x0 = (src[0       + j] - min)*id;
        स्थिरांक दशांश x1 = (src[QK5_1/2 + j] - min)*id;

        स्थिरांक uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        स्थिरांक uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread स्थिरांक uint8_t * qh8 = (thread स्थिरांक uint8_t *)&qh;

    साठी (पूर्णांक j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

शून्य quantize_q8_0(device स्थिरांक दशांश * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    दशांश amax = 0.0f; // absolute max

    साठी (पूर्णांक j = 0; j < QK8_0; j++) {
        स्थिरांक दशांश v = src[j];
        amax = MAX(amax, भ_असल(v));
    }

    स्थिरांक दशांश d = amax / ((1 << 7) - 1);
    स्थिरांक दशांश id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    साठी (पूर्णांक j = 0; j < QK8_0; ++j) {
        स्थिरांक दशांश x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

शून्य quantize_iq4_nl(device स्थिरांक दशांश * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    दशांश amax = 0.0f; // absolute max
    दशांश max  = 0.0f;

    साठी (पूर्णांक j = 0; j < QK4_NL; j++) {
        स्थिरांक दशांश v = src[j];
        जर (amax < भ_असल(v)) {
            amax = भ_असल(v);
            max  = v;
        }
    }

    स्थिरांक दशांश d = max / kvalues_iq4nl_f[0];
    स्थिरांक दशांश id = d ? 1.0f/d : 0.0f;

    दशांश sumqx = 0, sumq2 = 0;
    साठी (पूर्णांक j = 0; j < QK4_NL/2; ++j) {
        स्थिरांक दशांश x0 = src[0        + j]*id;
        स्थिरांक दशांश x1 = src[QK4_NL/2 + j]*id;

        स्थिरांक uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        स्थिरांक uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        स्थिरांक दशांश v0 = kvalues_iq4nl_f[xi0];
        स्थिरांक दशांश v1 = kvalues_iq4nl_f[xi1];
        स्थिरांक दशांश w0 = src[0        + j]*src[0        + j];
        स्थिरांक दशांश w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

शून्य quantize_tq2_0(device स्थिरांक दशांश * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    दशांश amax = 0.0f; // absolute max

    साठी (पूर्णांक j = 0; j < QK_K; j++) {
        स्थिरांक दशांश v = src[j];
        amax = MAX(amax, भ_असल(v));
    }

    स्थिरांक दशांश d = amax;
    स्थिरांक दशांश id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    साठी (पूर्णांक j = 0; j < QK_K/4; j += 32) {
        साठी (पूर्णांक m = 0; m < 32; ++m) {
            uint8_t q = 0;
            साठी (पूर्णांक n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                पूर्णांक xi = (पूर्णांक)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
