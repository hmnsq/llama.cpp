#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "arch-fallback.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h> // for qsort
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>  // for GGML_ASSERT

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS 1e-15f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ3_XXS 1e-8f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ2_S 1e-8f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ1_M 1e-7f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ1_S 1e-12f

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ GGML_ਅਣਵਰਤਿਆ

ਖਾਲੀ quantize_ਪੰਕਤੀ_q1_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q1_0_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q2_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q2_0_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q4_0_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q4_1_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q5_0_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q5_1_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_0_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q8_0_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_1_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q8_1_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_mxfp4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_mxfp4_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_nvfp4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_nvfp4_ref(x, y, k);
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

ਖਾਲੀ quantize_ਪੰਕਤੀ_q2_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    quantize_ਪੰਕਤੀ_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

ਖਾਲੀ quantize_ਪੰਕਤੀ_q3_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    quantize_ਪੰਕਤੀ_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਖੰਡ_q4_K * GGML_RESTRICT y = vy;
    quantize_ਪੰਕਤੀ_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਖੰਡ_q5_K * GGML_RESTRICT y = vy;
    quantize_ਪੰਕਤੀ_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

ਖਾਲੀ quantize_ਪੰਕਤੀ_q6_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਖੰਡ_q6_K * GGML_RESTRICT y = vy;
    quantize_ਪੰਕਤੀ_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

ਖਾਲੀ quantize_ਪੰਕਤੀ_tq1_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਖੰਡ_tq1_0 * GGML_RESTRICT y = vy;
    quantize_ਪੰਕਤੀ_tq1_0_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_tq2_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    ਖੰਡ_tq2_0 * GGML_RESTRICT y = vy;
    quantize_ਪੰਕਤੀ_tq2_0_ref(x, y, k);
}

//===================================== Q8_K ==============================================

ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_K_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    quantize_ਪੰਕਤੀ_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

ਖਾਲੀ ggml_vec_dot_q1_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK1_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q1_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਦਸ਼ਮਲਵ sumf = 0.0;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਅਚਲ ਦਸ਼ਮਲਵ d0 = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d);

        ਦਸ਼ਮਲਵ sumi = 0.0f;

        ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
            ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(yb->d);
            ਪੂਰਨ sumi_ਖੰਡ = 0;

            ਅਚਲ uint8_t * GGML_RESTRICT ਬਿੱਟ = &x[i].qs[k * 4];
            ਅਚਲ int8_t  * GGML_RESTRICT qy   = yb->qs;

            ਲਈ (ਪੂਰਨ b = 0; b < 4; ++b, qy += 8) {
                ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਨਕਾਬ = ਬਿੱਟ[b];
                sumi_ਖੰਡ += ((ਨਕਾਬ & 0x01) ? qy[0] : -qy[0])
                           +  ((ਨਕਾਬ & 0x02) ? qy[1] : -qy[1])
                           +  ((ਨਕਾਬ & 0x04) ? qy[2] : -qy[2])
                           +  ((ਨਕਾਬ & 0x08) ? qy[3] : -qy[3])
                           +  ((ਨਕਾਬ & 0x10) ? qy[4] : -qy[4])
                           +  ((ਨਕਾਬ & 0x20) ? qy[5] : -qy[5])
                           +  ((ਨਕਾਬ & 0x40) ? qy[6] : -qy[6])
                           +  ((ਨਕਾਬ & 0x80) ? qy[7] : -qy[7]);
            }

            sumi += d1 * sumi_ਖੰਡ;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q2_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK2_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q2_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਦਸ਼ਮਲਵ sumf = 0.0f;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਅਚਲ ਦਸ਼ਮਲਵ d0 = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d);

        ਦਸ਼ਮਲਵ sumi = 0.0f;

        // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
        ਲਈ (ਪੂਰਨ k = 0; k < 2; k++) {
            ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT yb = &y[i * 2 + k];
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(yb->d);
            ਪੂਰਨ sumi_ਖੰਡ = 0;

            ਅਚਲ uint8_t * GGML_RESTRICT qs = &x[i].qs[k * 8];
            ਅਚਲ int8_t  * GGML_RESTRICT qy = yb->qs;

            ਲਈ (ਪੂਰਨ b = 0; b < 8; ++b) {
                ਅਚਲ uint8_t byte = qs[b];
                // Extract 4 two-bit values, map {0,1,2,3} -> {-1,0,1,2}
                sumi_ਖੰਡ += ((ਪੂਰਨ)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
                sumi_ਖੰਡ += ((ਪੂਰਨ)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
                sumi_ਖੰਡ += ((ਪੂਰਨ)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
                sumi_ਖੰਡ += ((ਪੂਰਨ)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
            }

            sumi += d1 * sumi_ਖੰਡ;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q4_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q4_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਲਈ (; ib < nb; ++ib) {
        ਪੂਰਨ sumi0 = 0;
        ਪੂਰਨ sumi1 = 0;

        ਲਈ (ਪੂਰਨ j = 0; j < qk/2; ++j) {
            ਅਚਲ ਪੂਰਨ v0 = (x[ib].qs[j] & 0x0F) - 8;
            ਅਚਲ ਪੂਰਨ v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        ਪੂਰਨ sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
ਖਾਲੀ ggml_vec_dot_q4_1_q8_1_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_1;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q4_1 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_1 * GGML_RESTRICT y = vy;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਲਈ (; ib < nb; ++ib) {
        ਪੂਰਨ sumi0 = 0;
        ਪੂਰਨ sumi1 = 0;

        ਲਈ (ਪੂਰਨ j = 0; j < qk/2; ++j) {
            ਅਚਲ ਪੂਰਨ v0 = (x[ib].qs[j] & 0x0F);
            ਅਚਲ ਪੂਰਨ v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        ਪੂਰਨ sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d))*sumi + GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].m)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].s);
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_mxfp4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ਅਚਲ ਖੰਡ_mxfp4 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_MXFP4;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਲਈ (; ib < nb; ++ib) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d)*GGML_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(x[ib].e);

        ਪੂਰਨ sumi1 = 0;
        ਪੂਰਨ sumi2 = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
ਖਾਲੀ ggml_vec_dot_nvfp4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_NVFP4 == 0);

    ਅਚਲ ਖੰਡ_nvfp4 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_NVFP4;

    ਦਸ਼ਮਲਵ sumf = 0;

    ਲਈ (ਪੂਰਨ ib = 0; ib < nb; ++ib) {
        ਲਈ (ਪੂਰਨ s_idx = 0; s_idx < 4; ++s_idx) {
            ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_ue4m3_ਮੰਜ਼ਿਲ_fp32(x[ib].d[s_idx]);
            ਅਚਲ ਪੂਰਨ q8_ਖੰਡ = s_idx / 2;
            ਅਚਲ ਪੂਰਨ q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            ਅਚਲ ਦਸ਼ਮਲਵ dy = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[2*ib + q8_ਖੰਡ].d);

            ਪੂਰਨ sumi_lo = 0, sumi_hi = 0;
            ਲਈ (ਪੂਰਨ j = 0; j < QK_NVFP4_SUB/2; ++j) {
                ਅਚਲ uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_ਖੰਡ].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_ਖੰਡ].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q5_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(qk == QK5_0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q5_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਲਈ (; ib < nb; ++ib) {
        uint32_t qh;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&qh, x[ib].qh, ਆਕਾਰ(qh));

        ਪੂਰਨ sumi0 = 0;
        ਪੂਰਨ sumi1 = 0;

        ਲਈ (ਪੂਰਨ j = 0; j < qk/2; ++j) {
            ਅਚਲ uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            ਅਚਲ uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            ਅਚਲ int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            ਅਚਲ int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        ਪੂਰਨ sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q5_1_q8_1_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_1;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(qk == QK5_1);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q5_1 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_1 * GGML_RESTRICT y = vy;

    ਲਈ (; ib < nb; ++ib) {
        uint32_t qh;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&qh, x[ib].qh, ਆਕਾਰ(qh));

        ਪੂਰਨ sumi0 = 0;
        ਪੂਰਨ sumi1 = 0;

        ਲਈ (ਪੂਰਨ j = 0; j < qk/2; ++j) {
            ਅਚਲ uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            ਅਚਲ uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            ਅਚਲ int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            ਅਚਲ int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        ਪੂਰਨ sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d))*sumi + GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].m)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].s);
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q8_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਲਈ (; ib < nb; ++ib) {
        ਪੂਰਨ sumi = 0;

        ਲਈ (ਪੂਰਨ j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d));
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_tq1_0_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_tq1_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਅਚਲ uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    ਦਸ਼ਮਲਵ sumf = 0.0f;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਪੂਰਨ ਜੋੜ = 0;

        ਲਈ (size_t j = 0; j < ਆਕਾਰ(x->qs) - ਆਕਾਰ(x->qs) % 32; j += 32) {
            ਲਈ (size_t l = 0; l < 5; ++l) {
                ਲਈ (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    ਜੋੜ += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        ਲਈ (size_t j = ਆਕਾਰ(x->qs) - ਆਕਾਰ(x->qs) % 32; j < ਆਕਾਰ(x->qs); j += 16) {
            ਲਈ (size_t l = 0; l < 5; ++l) {
                ਲਈ (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    ਜੋੜ += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        ਲਈ (size_t l = 0; l < 4; ++l) {
            ਲਈ (size_t j = 0; j < ਆਕਾਰ(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                ਜੋੜ += (xi - 1) * y[i].qs[ਆਕਾਰ(x->qs)*5 + l*ਆਕਾਰ(x->qh) + j];
            }
        }

        sumf += (ਦਸ਼ਮਲਵ) ਜੋੜ * (GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_tq2_0_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_tq2_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;
    ਦਸ਼ਮਲਵ sumf = 0.0f;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        ਲਈ (size_t j = 0; j < ਆਕਾਰ(x->qs); j += 32) {
            ਲਈ (size_t l = 0; l < 4; ++l) {
                ਲਈ (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        ਅਚਲ ਦਸ਼ਮਲਵ d = y[i].d * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d);

        sumf += (ਦਸ਼ਮਲਵ) sumi * d;
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q2_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q2_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਦਸ਼ਮਲਵ sumf = 0;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {

        ਅਚਲ uint8_t * q2 = x[i].qs;
        ਅਚਲ  int8_t * q8 = y[i].qs;
        ਅਚਲ uint8_t * sc = x[i].scales;

        ਪੂਰਨ summs = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        ਅਚਲ ਦਸ਼ਮਲਵ dall = y[i].d * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d);
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = y[i].d * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].dmin);

        ਪੂਰਨ isum = 0;
        ਪੂਰਨ is = 0;
        ਪੂਰਨ d;
        ਲਈ (ਪੂਰਨ k = 0; k < QK_K/128; ++k) {
            ਪੂਰਨ ਖਿਸਕਾਓ = 0;
            ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                ਪੂਰਨ isuml = 0;
                ਲਈ (ਪੂਰਨ l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> ਖਿਸਕਾਓ) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                ਲਈ (ਪੂਰਨ l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> ਖਿਸਕਾਓ) & 3);
                isum += d * isuml;
                ਖਿਸਕਾਓ += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q3_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ uint32_t kmask1 = 0x03030303;
    ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;

    ਅਚਲ ਖੰਡ_q3_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    ਦਸ਼ਮਲਵ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    uint32_t auxs[4];
    ਅਚਲ int8_t * scales = (ਅਚਲ int8_t*)auxs;

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ਅਚਲ uint8_t * GGML_RESTRICT hm = x[i].hmask;
        ਅਚਲ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*ਆਕਾਰ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j += 128) {
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        ਸਮ੍ਰਿਤੀ_ਨਕਲ(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/16; ++j) {
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q4_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q4_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    ਅਚਲ uint8_t * scales = (ਅਚਲ uint8_t*)&utmp[0];
    ਅਚਲ uint8_t * mins   = (ਅਚਲ uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    ਦਸ਼ਮਲਵ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ਅਚਲ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*ਆਕਾਰ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/64; ++j) {
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ਅਚਲ uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        ਪੂਰਨ sumi = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        ਪੂਰਨ is = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q5_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,  size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q5_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    ਅਚਲ uint8_t * scales = (ਅਚਲ uint8_t*)&utmp[0];
    ਅਚਲ uint8_t * mins   = (ਅਚਲ uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    ਦਸ਼ਮਲਵ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ਅਚਲ uint8_t * GGML_RESTRICT hm = x[i].qh;
        ਅਚਲ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*ਆਕਾਰ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/64; ++j) {
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        ਅਚਲ uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        ਪੂਰਨ sumi = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        ਪੂਰਨ is = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_q6_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q6_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    ਦਸ਼ਮਲਵ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ uint8_t * GGML_RESTRICT q4 = x[i].ql;
        ਅਚਲ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ਅਚਲ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*ਆਕਾਰ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j += 128) {
            ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        ਪੂਰਨ is = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/16; ++j) {
            ਪੂਰਨ scale = x[i].scales[is++];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    ਲਈ (ਪੂਰਨ l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_iq2_xxs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq2_xxs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K    * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    uint32_t aux32[2];
    ਅਚਲ uint8_t * aux8 = (ਅਚਲ uint8_t *)aux32;

    ਦਸ਼ਮਲਵ sumf = 0.f;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਅਚਲ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ਅਚਲ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        ਲਈ (ਪੂਰਨ ib32 = 0; ib32 < QK_K/32; ++ib32) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(aux32, q2, 2*ਆਕਾਰ(uint32_t));
            q2 += 4;
            ਅਚਲ uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *)(iq2xxs_grid + aux8[l]);
                ਅਚਲ uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

ਖਾਲੀ ggml_vec_dot_iq2_xs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq2_xs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K   * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਦਸ਼ਮਲਵ sumf = 0.f;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਅਚਲ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ਅਚਲ uint8_t  * GGML_RESTRICT sc = x[i].scales;
        ਅਚਲ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        ਲਈ (ਪੂਰਨ ib32 = 0; ib32 < QK_K/32; ++ib32) {
            ਅਚਲ uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            ਅਚਲ uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 2; ++l) {
                ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *)(iq2xs_grid + (q2[l] & 511));
                ਅਚਲ uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            ਲਈ (ਪੂਰਨ l = 2; l < 4; ++l) {
                ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *)(iq2xs_grid + (q2[l] & 511));
                ਅਚਲ uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

ਖਾਲੀ ggml_vec_dot_iq2_s_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq2_s * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {

        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਅਚਲ int8_t  * q8 = y[i].qs;
        ਅਚਲ uint8_t * qs = x[i].qs;
        ਅਚਲ uint8_t * qh = x[i].qh;
        ਅਚਲ uint8_t * signs = qs + QK_K/8;

        ਪੂਰਨ bsum = 0;
        ਲਈ (ਪੂਰਨ ib32 = 0; ib32 < QK_K/32; ++ib32) {
            ਪੂਰਨ ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            ਪੂਰਨ ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            ਪੂਰਨ sumi1 = 0, sumi2 = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 2; ++l) {
                ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            ਲਈ (ਪੂਰਨ l = 2; l < 4; ++l) {
                ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;
}

ਖਾਲੀ ggml_vec_dot_iq3_xxs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq3_xxs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K    * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    uint32_t aux32;

    ਦਸ਼ਮਲਵ sumf = 0.f;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਅਚਲ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ਅਚਲ uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        ਲਈ (ਪੂਰਨ ib32 = 0; ib32 < QK_K/32; ++ib32) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&aux32, gas, ਆਕਾਰ(uint32_t)); gas += ਆਕਾਰ(uint32_t);
            ਅਚਲ uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਅਚਲ uint8_t * grid1 = (ਅਚਲ uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                ਅਚਲ uint8_t * grid2 = (ਅਚਲ uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                ਅਚਲ uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

ਖਾਲੀ ggml_vec_dot_iq3_s_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq3_s * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਦਸ਼ਮਲਵ sumf = 0.f;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d;
        ਅਚਲ uint8_t * GGML_RESTRICT qs = x[i].qs;
        ਅਚਲ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ਅਚਲ uint8_t * GGML_RESTRICT signs = x[i].signs;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        ਲਈ (ਪੂਰਨ ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ਅਚਲ uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            ਅਚਲ uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਅਚਲ uint8_t * grid1 = (ਅਚਲ uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                ਅਚਲ uint8_t * grid2 = (ਅਚਲ uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਅਚਲ uint8_t * grid1 = (ਅਚਲ uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                ਅਚਲ uint8_t * grid2 = (ਅਚਲ uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_iq1_s_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq1_s * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {

        ਅਚਲ int8_t   * q8 = y[i].qs;
        ਅਚਲ uint8_t  * qs = x[i].qs;
        ਅਚਲ uint16_t * qh = x[i].qh;

        ਪੂਰਨ sumi = 0, sumi1 = 0;
        ਲਈ (ਪੂਰਨ ib = 0; ib < QK_K/32; ++ib) {
            ਅਚਲ ਪੂਰਨ ls = 2*((qh[ib] >> 12) & 7) + 1;
            ਅਚਲ ਪੂਰਨ delta = qh[ib] & 0x8000 ? -1 : 1;
            ਪੂਰਨ lsum = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਅਚਲ int8_t * grid = (ਅਚਲ int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_iq1_m_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq1_m * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    iq1m_scale_t scale;

    ਪੂਰਨ sum1[2], sum2[2], delta[4];

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {

        ਅਚਲ int8_t   * q8 = y[i].qs;
        ਅਚਲ uint8_t  * qs = x[i].qs;
        ਅਚਲ uint8_t  * qh = x[i].qh;
        ਅਚਲ uint16_t * sc = (ਅਚਲ uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        ਪੂਰਨ sumi1 = 0, sumi2 = 0;
        ਲਈ (ਪੂਰਨ ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਅਚਲ int8_t * grid = (ਅਚਲ int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                ਪੂਰਨ lsum1 = 0, lsum2 = 0;
                ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            ਅਚਲ ਪੂਰਨ ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            ਅਚਲ ਪੂਰਨ ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_iq4_nl_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ਅਚਲ ਖੰਡ_iq4_nl * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0   * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK4_NL;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਲਈ (; ib < nb; ++ib) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d)*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d);
        ਪੂਰਨ sumi1 = 0, sumi2 = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_iq4_xs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);

    ਅਚਲ ਖੰਡ_iq4_xs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K   * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਦਸ਼ਮਲਵ sumf = 0;
    ਲਈ (ਪੂਰਨ ibl = 0; ibl < nb; ++ibl) {
        ਅਚਲ ਦਸ਼ਮਲਵ d4d8 = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        ਅਚਲ uint8_t * qs = x[ibl].qs;
        ਅਚਲ int8_t  * q8 = y[ibl].qs;
        ਲਈ (ਪੂਰਨ ib = 0; ib < QK_K/32; ib += 2) {
            ਅਚਲ uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            ਅਚਲ uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            ਅਚਲ ਦਸ਼ਮਲਵ d1 = d4d8*(ls1 - 32);
            ਅਚਲ ਦਸ਼ਮਲਵ d2 = d4d8*(ls2 - 32);
            ਪੂਰਨ sumi1 = 0, sumi2 = 0;
            ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            ਲਈ (ਪੂਰਨ j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

// ============================ 4-bit non-linear quants

ਖਾਲੀ quantize_ਪੰਕਤੀ_iq4_nl(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK4_NL == 0);
    quantize_ਪੰਕਤੀ_iq4_nl_ref(x, y, k);
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_iq4_xs(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, ਨਿਰਾ);
}
