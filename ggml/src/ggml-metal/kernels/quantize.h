#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "common.h"

ਖਾਲੀ quantize_q1_0(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_q1_0 & dst) {
    ਦਸ਼ਮਲਵ ਜੋੜ_abs = 0.0f;
    ਲਈ (ਪੂਰਨ j = 0; j < QK1_0; j++) {
        ਜੋੜ_abs += fabs(src[j]);
    }
    dst.d = ਜੋੜ_abs / QK1_0;

    ਲਈ (ਪੂਰਨ j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    ਲਈ (ਪੂਰਨ j = 0; j < QK1_0; j++) {
        ਜੇ (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

ਖਾਲੀ quantize_q2_0(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_q2_0 & dst) {
    ਦਸ਼ਮਲਵ amax = 0.0f;
    ਲਈ (ਪੂਰਨ j = 0; j < QK2_0; j++) {
        ਦਸ਼ਮਲਵ a = fabs(src[j]);
        ਜੇ (a > amax) amax = a;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ d = amax;
    dst.d = d;

    ਅਚਲ ਦਸ਼ਮਲਵ id = d > 0.0f ? 1.0f / d : 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    ਲਈ (ਪੂਰਨ j = 0; j < QK2_0; j++) {
        ਪੂਰਨ q = (ਪੂਰਨ)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

ਖਾਲੀ quantize_q4_0(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_q4_0 & dst) {
#ਹਦਾਇਤ METAL fp math_ਬਹੁਲਕ(ਸੁਰੱਖਿਅਤ_2)
    ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max
    ਦਸ਼ਮਲਵ max  = 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_0; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = src[j];
        ਜੇ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = max / -8;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_0/2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = src[0       + j]*id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = src[QK4_0/2 + j]*id;

        ਅਚਲ uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        ਅਚਲ uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ਖਾਲੀ quantize_q4_1(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_q4_1 & dst) {
#ਹਦਾਇਤ METAL fp math_ਬਹੁਲਕ(ਸੁਰੱਖਿਅਤ_2)
    ਦਸ਼ਮਲਵ min = FLT_MAX;
    ਦਸ਼ਮਲਵ max = -FLT_MAX;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_1; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = src[j];
        ਜੇ (min > v) min = v;
        ਜੇ (max < v) max = v;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = (max - min) / ((1 << 4) - 1);
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_1/2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = (src[0       + j] - min)*id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = (src[QK4_1/2 + j] - min)*id;

        ਅਚਲ uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        ਅਚਲ uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ਖਾਲੀ quantize_q5_0(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_q5_0 & dst) {
#ਹਦਾਇਤ METAL fp math_ਬਹੁਲਕ(ਸੁਰੱਖਿਅਤ_2)
    ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max
    ਦਸ਼ਮਲਵ max  = 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK5_0; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = src[j];
        ਜੇ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = max / -16;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < QK5_0/2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = src[0       + j]*id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = src[QK5_0/2 + j]*id;

        ਅਚਲ uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        ਅਚਲ uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread ਅਚਲ uint8_t * qh8 = (thread ਅਚਲ uint8_t *)&qh;

    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ਖਾਲੀ quantize_q5_1(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_q5_1 & dst) {
#ਹਦਾਇਤ METAL fp math_ਬਹੁਲਕ(ਸੁਰੱਖਿਅਤ_2)
    ਦਸ਼ਮਲਵ max = src[0];
    ਦਸ਼ਮਲਵ min = src[0];

    ਲਈ (ਪੂਰਨ j = 1; j < QK5_1; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = (max - min) / 31;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < QK5_1/2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = (src[0       + j] - min)*id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = (src[QK5_1/2 + j] - min)*id;

        ਅਚਲ uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        ਅਚਲ uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread ਅਚਲ uint8_t * qh8 = (thread ਅਚਲ uint8_t *)&qh;

    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ਖਾਲੀ quantize_q8_0(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_q8_0 & dst) {
#ਹਦਾਇਤ METAL fp math_ਬਹੁਲਕ(ਸੁਰੱਖਿਅਤ_2)
    ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max

    ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = amax / ((1 << 7) - 1);
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

ਖਾਲੀ quantize_iq4_nl(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_iq4_nl & dst) {
#ਹਦਾਇਤ METAL fp math_ਬਹੁਲਕ(ਸੁਰੱਖਿਅਤ_2)
    ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max
    ਦਸ਼ਮਲਵ max  = 0.0f;

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_NL; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = src[j];
        ਜੇ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = max / kvalues_iq4nl_f[0];
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;

    ਦਸ਼ਮਲਵ sumqx = 0, sumq2 = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < QK4_NL/2; ++j) {
        ਅਚਲ ਦਸ਼ਮਲਵ x0 = src[0        + j]*id;
        ਅਚਲ ਦਸ਼ਮਲਵ x1 = src[QK4_NL/2 + j]*id;

        ਅਚਲ uint8_t xi0 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_int8(16, kvalues_iq4nl_f, x0);
        ਅਚਲ uint8_t xi1 = ਉੱਤਮ_ਸੂਚਕਾਂਕ_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        ਅਚਲ ਦਸ਼ਮਲਵ v0 = kvalues_iq4nl_f[xi0];
        ਅਚਲ ਦਸ਼ਮਲਵ v1 = kvalues_iq4nl_f[xi1];
        ਅਚਲ ਦਸ਼ਮਲਵ w0 = src[0        + j]*src[0        + j];
        ਅਚਲ ਦਸ਼ਮਲਵ w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

ਖਾਲੀ quantize_tq2_0(device ਅਚਲ ਦਸ਼ਮਲਵ * src, device ਖੰਡ_tq2_0 & dst) {
#ਹਦਾਇਤ METAL fp math_ਬਹੁਲਕ(ਸੁਰੱਖਿਅਤ_2)
    ਦਸ਼ਮਲਵ amax = 0.0f; // absolute max

    ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j++) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = src[j];
        amax = MAX(amax, fabs(v));
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = amax;
    ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    ਲਈ (ਪੂਰਨ j = 0; j < QK_K/4; j += 32) {
        ਲਈ (ਪੂਰਨ m = 0; m < 32; ++m) {
            uint8_t q = 0;
            ਲਈ (ਪੂਰਨ n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                ਪੂਰਨ xi = (ਪੂਰਨ)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
