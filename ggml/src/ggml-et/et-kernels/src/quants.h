// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#അന്തര്ഭാവ "math_fp.h"

#അന്തര്ഭാവ <stdint.h>

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

സ്ഥിതം വരിയിൽ ശൂന്യം dequantize_q8_0_block(സ്ഥിരം block_q8_0 * block, ദശാംശം * dst) {
    സ്ഥിരം ദശാംശം scale = fp16_to_fp32(block->d);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QK8_0; i++) {
        dst[i] = scale * (ദശാംശം) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
സ്ഥിതം വരിയിൽ ശൂന്യം dequantize_q4_0_block(സ്ഥിരം block_q4_0 * block, ദശാംശം * dst) {
    സ്ഥിരം ദശാംശം scale = fp16_to_fp32(block->d);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < QK4_0 / 2; i++) {
        സ്ഥിരം uint8_t byte = block->qs[i];
        dst[i]             = scale * (ദശാംശം) ((പൂർണ്ണസംഖ്യ) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (ദശാംശം) ((പൂർണ്ണസംഖ്യ) (byte >> 4) - 8);
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
സ്ഥിതം വരിയിൽ ശൂന്യം get_scale_min_k4(പൂർണ്ണസംഖ്യ j, സ്ഥിരം uint8_t * q, uint8_t * d, uint8_t * m) {
    എങ്കിൽ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } അല്ലെങ്കിൽ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം dequantize_q4_K_block(സ്ഥിരം block_q4_K * block, ദശാംശം * dst) {
    സ്ഥിരം uint8_t * q   = block->qs;
    സ്ഥിരം ദശാംശം     d   = fp16_to_fp32(block->d);
    സ്ഥിരം ദശാംശം     min = fp16_to_fp32(block->dmin);

    പൂർണ്ണസംഖ്യ     is = 0;
    uint8_t sc, m;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        സ്ഥിരം ദശാംശം d1 = d * sc;
        സ്ഥിരം ദശാംശം m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        സ്ഥിരം ദശാംശം d2 = d * sc;
        സ്ഥിരം ദശാംശം m2 = min * m;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

#endif  // QUANTS_H
