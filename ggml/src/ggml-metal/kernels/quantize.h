#pragma once

#അന്തര്ഭാവ "common.h"

ശൂന്യം quantize_q1_0(device സ്ഥിരം ദശാംശം * src, device block_q1_0 & dst) {
    ദശാംശം sum_abs = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK1_0; j++) {
        എങ്കിൽ (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

ശൂന്യം quantize_q2_0(device സ്ഥിരം ദശാംശം * src, device block_q2_0 & dst) {
    ദശാംശം amax = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK2_0; j++) {
        ദശാംശം a = fabs(src[j]);
        എങ്കിൽ (a > amax) amax = a;
    }
    സ്ഥിരം ദശാംശം d = amax;
    dst.d = d;

    സ്ഥിരം ദശാംശം id = d > 0.0f ? 1.0f / d : 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK2_0; j++) {
        പൂർണ്ണസംഖ്യ q = (പൂർണ്ണസംഖ്യ)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

ശൂന്യം quantize_q4_0(device സ്ഥിരം ദശാംശം * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    ദശാംശം amax = 0.0f; // absolute max
    ദശാംശം max  = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_0; j++) {
        സ്ഥിരം ദശാംശം v = src[j];
        എങ്കിൽ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    സ്ഥിരം ദശാംശം d = max / -8;
    സ്ഥിരം ദശാംശം id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_0/2; ++j) {
        സ്ഥിരം ദശാംശം x0 = src[0       + j]*id;
        സ്ഥിരം ദശാംശം x1 = src[QK4_0/2 + j]*id;

        സ്ഥിരം uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        സ്ഥിരം uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ശൂന്യം quantize_q4_1(device സ്ഥിരം ദശാംശം * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    ദശാംശം min = FLT_MAX;
    ദശാംശം max = -FLT_MAX;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_1; j++) {
        സ്ഥിരം ദശാംശം v = src[j];
        എങ്കിൽ (min > v) min = v;
        എങ്കിൽ (max < v) max = v;
    }

    സ്ഥിരം ദശാംശം d = (max - min) / ((1 << 4) - 1);
    സ്ഥിരം ദശാംശം id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_1/2; ++j) {
        സ്ഥിരം ദശാംശം x0 = (src[0       + j] - min)*id;
        സ്ഥിരം ദശാംശം x1 = (src[QK4_1/2 + j] - min)*id;

        സ്ഥിരം uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        സ്ഥിരം uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

ശൂന്യം quantize_q5_0(device സ്ഥിരം ദശാംശം * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    ദശാംശം amax = 0.0f; // absolute max
    ദശാംശം max  = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK5_0; j++) {
        സ്ഥിരം ദശാംശം v = src[j];
        എങ്കിൽ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    സ്ഥിരം ദശാംശം d = max / -16;
    സ്ഥിരം ദശാംശം id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK5_0/2; ++j) {
        സ്ഥിരം ദശാംശം x0 = src[0       + j]*id;
        സ്ഥിരം ദശാംശം x1 = src[QK5_0/2 + j]*id;

        സ്ഥിരം uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        സ്ഥിരം uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread സ്ഥിരം uint8_t * qh8 = (thread സ്ഥിരം uint8_t *)&qh;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ശൂന്യം quantize_q5_1(device സ്ഥിരം ദശാംശം * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    ദശാംശം max = src[0];
    ദശാംശം min = src[0];

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < QK5_1; j++) {
        സ്ഥിരം ദശാംശം v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    സ്ഥിരം ദശാംശം d = (max - min) / 31;
    സ്ഥിരം ദശാംശം id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK5_1/2; ++j) {
        സ്ഥിരം ദശാംശം x0 = (src[0       + j] - min)*id;
        സ്ഥിരം ദശാംശം x1 = (src[QK5_1/2 + j] - min)*id;

        സ്ഥിരം uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        സ്ഥിരം uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread സ്ഥിരം uint8_t * qh8 = (thread സ്ഥിരം uint8_t *)&qh;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

ശൂന്യം quantize_q8_0(device സ്ഥിരം ദശാംശം * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    ദശാംശം amax = 0.0f; // absolute max

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; j++) {
        സ്ഥിരം ദശാംശം v = src[j];
        amax = MAX(amax, fabs(v));
    }

    സ്ഥിരം ദശാംശം d = amax / ((1 << 7) - 1);
    സ്ഥിരം ദശാംശം id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; ++j) {
        സ്ഥിരം ദശാംശം x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

ശൂന്യം quantize_iq4_nl(device സ്ഥിരം ദശാംശം * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    ദശാംശം amax = 0.0f; // absolute max
    ദശാംശം max  = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_NL; j++) {
        സ്ഥിരം ദശാംശം v = src[j];
        എങ്കിൽ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    സ്ഥിരം ദശാംശം d = max / kvalues_iq4nl_f[0];
    സ്ഥിരം ദശാംശം id = d ? 1.0f/d : 0.0f;

    ദശാംശം sumqx = 0, sumq2 = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_NL/2; ++j) {
        സ്ഥിരം ദശാംശം x0 = src[0        + j]*id;
        സ്ഥിരം ദശാംശം x1 = src[QK4_NL/2 + j]*id;

        സ്ഥിരം uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        സ്ഥിരം uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        സ്ഥിരം ദശാംശം v0 = kvalues_iq4nl_f[xi0];
        സ്ഥിരം ദശാംശം v1 = kvalues_iq4nl_f[xi1];
        സ്ഥിരം ദശാംശം w0 = src[0        + j]*src[0        + j];
        സ്ഥിരം ദശാംശം w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

ശൂന്യം quantize_tq2_0(device സ്ഥിരം ദശാംശം * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    ദശാംശം amax = 0.0f; // absolute max

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K; j++) {
        സ്ഥിരം ദശാംശം v = src[j];
        amax = MAX(amax, fabs(v));
    }

    സ്ഥിരം ദശാംശം d = amax;
    സ്ഥിരം ദശാംശം id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K/4; j += 32) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ m = 0; m < 32; ++m) {
            uint8_t q = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                പൂർണ്ണസംഖ്യ xi = (പൂർണ്ണസംഖ്യ)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
