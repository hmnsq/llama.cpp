// Scalar dequantization helpers and ET-side block-size aliases.

#నిర్వచితం_కాకపోతే QUANTS_H
#నిర్వచించు QUANTS_H

#అన్తర్భావ "math_fp.h"

#అన్తర్భావ <stdint.h>

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#నిర్వచించు QK_F16 32
#నిర్వచించు QK_F32 16

స్థిర అంతఃవరుస శూన్య dequantize_q8_0_ఖండం(స్థిరరాశి ఖండం_q8_0 * ఖండం, దశాంశం * dst) {
    స్థిరరాశి దశాంశం scale = fp16_గమ్యం_fp32(ఖండం->d);

    కొరకు (పూర్ణాంకం i = 0; i < QK8_0; i++) {
        dst[i] = scale * (దశాంశం) ఖండం->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
స్థిర అంతఃవరుస శూన్య dequantize_q4_0_ఖండం(స్థిరరాశి ఖండం_q4_0 * ఖండం, దశాంశం * dst) {
    స్థిరరాశి దశాంశం scale = fp16_గమ్యం_fp32(ఖండం->d);

    కొరకు (పూర్ణాంకం i = 0; i < QK4_0 / 2; i++) {
        స్థిరరాశి uint8_t byte = ఖండం->qs[i];
        dst[i]             = scale * (దశాంశం) ((పూర్ణాంకం) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (దశాంశం) ((పూర్ణాంకం) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
స్థిర అంతఃవరుస శూన్య get_scale_min_k4(పూర్ణాంకం j, స్థిరరాశి uint8_t * q, uint8_t * d, uint8_t * m) {
    ఒకవేళ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } లేకపోతే {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

స్థిర అంతఃవరుస శూన్య dequantize_q4_K_ఖండం(స్థిరరాశి ఖండం_q4_K * ఖండం, దశాంశం * dst) {
    స్థిరరాశి uint8_t * q   = ఖండం->qs;
    స్థిరరాశి దశాంశం     d   = fp16_గమ్యం_fp32(ఖండం->d);
    స్థిరరాశి దశాంశం     min = fp16_గమ్యం_fp32(ఖండం->dmin);

    పూర్ణాంకం     is = 0;
    uint8_t sc, m;
    కొరకు (పూర్ణాంకం j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, ఖండం->scales, &sc, &m);
        స్థిరరాశి దశాంశం d1 = d * sc;
        స్థిరరాశి దశాంశం m1 = min * m;
        get_scale_min_k4(is + 1, ఖండం->scales, &sc, &m);
        స్థిరరాశి దశాంశం d2 = d * sc;
        స్థిరరాశి దశాంశం m2 = min * m;
        కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#అయితే_ముగింపు  // QUANTS_H
