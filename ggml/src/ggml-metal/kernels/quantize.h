#సూచన once

#అన్తర్భావ "common.h"

శూన్య quantize_q1_0(device స్థిరరాశి దశాంశం * src, device ఖండం_q1_0 & dst) {
    దశాంశం కూడిక_abs = 0.0f;
    కొరకు (పూర్ణాంకం j = 0; j < QK1_0; j++) {
        కూడిక_abs += fabs(src[j]);
    }
    dst.d = కూడిక_abs / QK1_0;

    కొరకు (పూర్ణాంకం j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    కొరకు (పూర్ణాంకం j = 0; j < QK1_0; j++) {
        ఒకవేళ (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

శూన్య quantize_q2_0(device స్థిరరాశి దశాంశం * src, device ఖండం_q2_0 & dst) {
    దశాంశం amax = 0.0f;
    కొరకు (పూర్ణాంకం j = 0; j < QK2_0; j++) {
        దశాంశం a = fabs(src[j]);
        ఒకవేళ (a > amax) amax = a;
    }
    స్థిరరాశి దశాంశం d = amax;
    dst.d = d;

    స్థిరరాశి దశాంశం id = d > 0.0f ? 1.0f / d : 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    కొరకు (పూర్ణాంకం j = 0; j < QK2_0; j++) {
        పూర్ణాంకం q = (పూర్ణాంకం)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

శూన్య quantize_q4_0(device స్థిరరాశి దశాంశం * src, device ఖండం_q4_0 & dst) {
#సూచన METAL fp math_బహుళకం(సురక్షిత)
    దశాంశం amax = 0.0f; // absolute max
    దశాంశం max  = 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_0; j++) {
        స్థిరరాశి దశాంశం v = src[j];
        ఒకవేళ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    స్థిరరాశి దశాంశం d = max / -8;
    స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_0/2; ++j) {
        స్థిరరాశి దశాంశం x0 = src[0       + j]*id;
        స్థిరరాశి దశాంశం x1 = src[QK4_0/2 + j]*id;

        స్థిరరాశి uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        స్థిరరాశి uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

శూన్య quantize_q4_1(device స్థిరరాశి దశాంశం * src, device ఖండం_q4_1 & dst) {
#సూచన METAL fp math_బహుళకం(సురక్షిత)
    దశాంశం min = FLT_MAX;
    దశాంశం max = -FLT_MAX;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_1; j++) {
        స్థిరరాశి దశాంశం v = src[j];
        ఒకవేళ (min > v) min = v;
        ఒకవేళ (max < v) max = v;
    }

    స్థిరరాశి దశాంశం d = (max - min) / ((1 << 4) - 1);
    స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_1/2; ++j) {
        స్థిరరాశి దశాంశం x0 = (src[0       + j] - min)*id;
        స్థిరరాశి దశాంశం x1 = (src[QK4_1/2 + j] - min)*id;

        స్థిరరాశి uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        స్థిరరాశి uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

శూన్య quantize_q5_0(device స్థిరరాశి దశాంశం * src, device ఖండం_q5_0 & dst) {
#సూచన METAL fp math_బహుళకం(సురక్షిత)
    దశాంశం amax = 0.0f; // absolute max
    దశాంశం max  = 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK5_0; j++) {
        స్థిరరాశి దశాంశం v = src[j];
        ఒకవేళ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    స్థిరరాశి దశాంశం d = max / -16;
    స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    కొరకు (పూర్ణాంకం j = 0; j < QK5_0/2; ++j) {
        స్థిరరాశి దశాంశం x0 = src[0       + j]*id;
        స్థిరరాశి దశాంశం x1 = src[QK5_0/2 + j]*id;

        స్థిరరాశి uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        స్థిరరాశి uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread స్థిరరాశి uint8_t * qh8 = (thread స్థిరరాశి uint8_t *)&qh;

    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

శూన్య quantize_q5_1(device స్థిరరాశి దశాంశం * src, device ఖండం_q5_1 & dst) {
#సూచన METAL fp math_బహుళకం(సురక్షిత)
    దశాంశం max = src[0];
    దశాంశం min = src[0];

    కొరకు (పూర్ణాంకం j = 1; j < QK5_1; j++) {
        స్థిరరాశి దశాంశం v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    స్థిరరాశి దశాంశం d = (max - min) / 31;
    స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    కొరకు (పూర్ణాంకం j = 0; j < QK5_1/2; ++j) {
        స్థిరరాశి దశాంశం x0 = (src[0       + j] - min)*id;
        స్థిరరాశి దశాంశం x1 = (src[QK5_1/2 + j] - min)*id;

        స్థిరరాశి uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        స్థిరరాశి uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread స్థిరరాశి uint8_t * qh8 = (thread స్థిరరాశి uint8_t *)&qh;

    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

శూన్య quantize_q8_0(device స్థిరరాశి దశాంశం * src, device ఖండం_q8_0 & dst) {
#సూచన METAL fp math_బహుళకం(సురక్షిత)
    దశాంశం amax = 0.0f; // absolute max

    కొరకు (పూర్ణాంకం j = 0; j < QK8_0; j++) {
        స్థిరరాశి దశాంశం v = src[j];
        amax = MAX(amax, fabs(v));
    }

    స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
    స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    కొరకు (పూర్ణాంకం j = 0; j < QK8_0; ++j) {
        స్థిరరాశి దశాంశం x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

శూన్య quantize_iq4_nl(device స్థిరరాశి దశాంశం * src, device ఖండం_iq4_nl & dst) {
#సూచన METAL fp math_బహుళకం(సురక్షిత)
    దశాంశం amax = 0.0f; // absolute max
    దశాంశం max  = 0.0f;

    కొరకు (పూర్ణాంకం j = 0; j < QK4_NL; j++) {
        స్థిరరాశి దశాంశం v = src[j];
        ఒకవేళ (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    స్థిరరాశి దశాంశం d = max / kvalues_iq4nl_f[0];
    స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

    దశాంశం sumqx = 0, sumq2 = 0;
    కొరకు (పూర్ణాంకం j = 0; j < QK4_NL/2; ++j) {
        స్థిరరాశి దశాంశం x0 = src[0        + j]*id;
        స్థిరరాశి దశాంశం x1 = src[QK4_NL/2 + j]*id;

        స్థిరరాశి uint8_t xi0 = ఉత్తమ_సూచీ_int8(16, kvalues_iq4nl_f, x0);
        స్థిరరాశి uint8_t xi1 = ఉత్తమ_సూచీ_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        స్థిరరాశి దశాంశం v0 = kvalues_iq4nl_f[xi0];
        స్థిరరాశి దశాంశం v1 = kvalues_iq4nl_f[xi1];
        స్థిరరాశి దశాంశం w0 = src[0        + j]*src[0        + j];
        స్థిరరాశి దశాంశం w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

శూన్య quantize_tq2_0(device స్థిరరాశి దశాంశం * src, device ఖండం_tq2_0 & dst) {
#సూచన METAL fp math_బహుళకం(సురక్షిత)
    దశాంశం amax = 0.0f; // absolute max

    కొరకు (పూర్ణాంకం j = 0; j < QK_K; j++) {
        స్థిరరాశి దశాంశం v = src[j];
        amax = MAX(amax, fabs(v));
    }

    స్థిరరాశి దశాంశం d = amax;
    స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    కొరకు (పూర్ణాంకం j = 0; j < QK_K/4; j += 32) {
        కొరకు (పూర్ణాంకం m = 0; m < 32; ++m) {
            uint8_t q = 0;
            కొరకు (పూర్ణాంకం n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                పూర్ణాంకం xi = (పూర్ణాంకం)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
