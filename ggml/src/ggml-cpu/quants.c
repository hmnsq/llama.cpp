#define GGML_COMMON_IMPL_C
#અન્તર્ભાવ "ggml-common.h"

#અન્તર્ભાવ "ggml-cpu-impl.h"
#અન્તર્ભાવ "simd-mappings.h"
#અન્તર્ભાવ "ggml-quants.h"
#અન્તર્ભાવ "quants.h"

#અન્તર્ભાવ "arch-fallback.h"

#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <float.h>
#અન્તર્ભાવ <stdlib.h> // for qsort
#અન્તર્ભાવ <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

શૂન્ય quantize_row_q1_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, y, k);
}

શૂન્ય quantize_row_q2_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q2_0_ref(x, y, k);
}

શૂન્ય quantize_row_q4_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

શૂન્ય quantize_row_q4_1(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

શૂન્ય quantize_row_q5_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

શૂન્ય quantize_row_q5_1(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

શૂન્ય quantize_row_q8_0_generic(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

શૂન્ય quantize_row_q8_1_generic(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}

શૂન્ય quantize_row_mxfp4(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_mxfp4_ref(x, y, k);
}

શૂન્ય quantize_row_nvfp4(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_nvfp4_ref(x, y, k);
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

શૂન્ય quantize_row_q2_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

શૂન્ય quantize_row_q3_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

શૂન્ય quantize_row_q4_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    પ્રતિજ્ઞા(k % QK_K == 0);
    block_q4_K * GGML_RESTRICT y = vy;
    quantize_row_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

શૂન્ય quantize_row_q5_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    પ્રતિજ્ઞા(k % QK_K == 0);
    block_q5_K * GGML_RESTRICT y = vy;
    quantize_row_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

શૂન્ય quantize_row_q6_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    પ્રતિજ્ઞા(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

શૂન્ય quantize_row_tq1_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    પ્રતિજ્ઞા(k % QK_K == 0);
    block_tq1_0 * GGML_RESTRICT y = vy;
    quantize_row_tq1_0_ref(x, y, k);
}

શૂન્ય quantize_row_tq2_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k) {
    પ્રતિજ્ઞા(k % QK_K == 0);
    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

//===================================== Q8_K ==============================================

શૂન્ય quantize_row_q8_K_generic(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

શૂન્ય ggml_vec_dot_q1_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    અચલ પૂર્ણાંક qk = QK1_0;
    અચલ પૂર્ણાંક nb = n / qk;

    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q1_0 * GGML_RESTRICT x = vx;
    અચલ block_q8_0 * GGML_RESTRICT y = vy;

    દશાંશ sumf = 0.0;

    માટે (પૂર્ણાંક i = 0; i < nb; i++) {
        અચલ દશાંશ d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        દશાંશ sumi = 0.0f;

        માટે (પૂર્ણાંક k = 0; k < 4; k++) {
            અચલ block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            અચલ દશાંશ d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            પૂર્ણાંક sumi_block = 0;

            અચલ uint8_t * GGML_RESTRICT bits = &x[i].qs[k * 4];
            અચલ int8_t  * GGML_RESTRICT qy   = yb->qs;

            માટે (પૂર્ણાંક b = 0; b < 4; ++b, qy += 8) {
                અચલ unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_q2_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    અચલ પૂર્ણાંક qk = QK2_0;
    અચલ પૂર્ણાંક nb = n / qk;

    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q2_0 * GGML_RESTRICT x = vx;
    અચલ block_q8_0 * GGML_RESTRICT y = vy;

    દશાંશ sumf = 0.0f;

    માટે (પૂર્ણાંક i = 0; i < nb; i++) {
        અચલ દશાંશ d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        દશાંશ sumi = 0.0f;

        // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
        માટે (પૂર્ણાંક k = 0; k < 2; k++) {
            અચલ block_q8_0 * GGML_RESTRICT yb = &y[i * 2 + k];
            અચલ દશાંશ d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            પૂર્ણાંક sumi_block = 0;

            અચલ uint8_t * GGML_RESTRICT qs = &x[i].qs[k * 8];
            અચલ int8_t  * GGML_RESTRICT qy = yb->qs;

            માટે (પૂર્ણાંક b = 0; b < 8; ++b) {
                અચલ uint8_t byte = qs[b];
                // Extract 4 two-bit values, map {0,1,2,3} -> {-1,0,1,2}
                sumi_block += ((પૂર્ણાંક)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
                sumi_block += ((પૂર્ણાંક)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
                sumi_block += ((પૂર્ણાંક)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
                sumi_block += ((પૂર્ણાંક)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_q4_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    અચલ પૂર્ણાંક qk = QK8_0;
    અચલ પૂર્ણાંક nb = n / qk;

    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q4_0 * GGML_RESTRICT x = vx;
    અચલ block_q8_0 * GGML_RESTRICT y = vy;

    પૂર્ણાંક ib = 0;
    દશાંશ sumf = 0;

    માટે (; ib < nb; ++ib) {
        પૂર્ણાંક sumi0 = 0;
        પૂર્ણાંક sumi1 = 0;

        માટે (પૂર્ણાંક j = 0; j < qk/2; ++j) {
            અચલ પૂર્ણાંક v0 = (x[ib].qs[j] & 0x0F) - 8;
            અચલ પૂર્ણાંક v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        પૂર્ણાંક sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
શૂન્ય ggml_vec_dot_q4_1_q8_1_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    અચલ પૂર્ણાંક qk = QK8_1;
    અચલ પૂર્ણાંક nb = n / qk;

    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q4_1 * GGML_RESTRICT x = vx;
    અચલ block_q8_1 * GGML_RESTRICT y = vy;

    પૂર્ણાંક ib = 0;
    દશાંશ sumf = 0;

    માટે (; ib < nb; ++ib) {
        પૂર્ણાંક sumi0 = 0;
        પૂર્ણાંક sumi1 = 0;

        માટે (પૂર્ણાંક j = 0; j < qk/2; ++j) {
            અચલ પૂર્ણાંક v0 = (x[ib].qs[j] & 0x0F);
            અચલ પૂર્ણાંક v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        પૂર્ણાંક sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_mxfp4_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    પ્રતિજ્ઞા(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    અચલ block_mxfp4 * GGML_RESTRICT x = vx;
    અચલ block_q8_0 * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_MXFP4;

    પૂર્ણાંક ib = 0;
    દશાંશ sumf = 0;

    માટે (; ib < nb; ++ib) {
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);

        પૂર્ણાંક sumi1 = 0;
        પૂર્ણાંક sumi2 = 0;
        માટે (પૂર્ણાંક j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
શૂન્ય ggml_vec_dot_nvfp4_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    પ્રતિજ્ઞા(n % QK_NVFP4 == 0);

    અચલ block_nvfp4 * GGML_RESTRICT x = vx;
    અચલ block_q8_0 * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_NVFP4;

    દશાંશ sumf = 0;

    માટે (પૂર્ણાંક ib = 0; ib < nb; ++ib) {
        માટે (પૂર્ણાંક s_idx = 0; s_idx < 4; ++s_idx) {
            અચલ દશાંશ d = ggml_ue4m3_to_fp32(x[ib].d[s_idx]);
            અચલ પૂર્ણાંક q8_block = s_idx / 2;
            અચલ પૂર્ણાંક q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            અચલ દશાંશ dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            પૂર્ણાંક sumi_lo = 0, sumi_hi = 0;
            માટે (પૂર્ણાંક j = 0; j < QK_NVFP4_SUB/2; ++j) {
                અચલ uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

શૂન્ય ggml_vec_dot_q5_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    અચલ પૂર્ણાંક qk = QK8_0;
    અચલ પૂર્ણાંક nb = n / qk;

    પૂર્ણાંક ib = 0;
    દશાંશ sumf = 0;

    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(qk == QK5_0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q5_0 * GGML_RESTRICT x = vx;
    અચલ block_q8_0 * GGML_RESTRICT y = vy;

    માટે (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, કદ(qh));

        પૂર્ણાંક sumi0 = 0;
        પૂર્ણાંક sumi1 = 0;

        માટે (પૂર્ણાંક j = 0; j < qk/2; ++j) {
            અચલ uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            અચલ uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            અચલ int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            અચલ int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        પૂર્ણાંક sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_q5_1_q8_1_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    અચલ પૂર્ણાંક qk = QK8_1;
    અચલ પૂર્ણાંક nb = n / qk;

    પૂર્ણાંક ib = 0;
    દશાંશ sumf = 0;

    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(qk == QK5_1);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q5_1 * GGML_RESTRICT x = vx;
    અચલ block_q8_1 * GGML_RESTRICT y = vy;

    માટે (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, કદ(qh));

        પૂર્ણાંક sumi0 = 0;
        પૂર્ણાંક sumi1 = 0;

        માટે (પૂર્ણાંક j = 0; j < qk/2; ++j) {
            અચલ uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            અચલ uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            અચલ int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            અચલ int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        પૂર્ણાંક sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_q8_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    અચલ પૂર્ણાંક qk = QK8_0;
    અચલ પૂર્ણાંક nb = n / qk;

    પ્રતિજ્ઞા(n % qk == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q8_0 * GGML_RESTRICT x = vx;
    અચલ block_q8_0 * GGML_RESTRICT y = vy;

    પૂર્ણાંક ib = 0;
    દશાંશ sumf = 0;

    માટે (; ib < nb; ++ib) {
        પૂર્ણાંક sumi = 0;

        માટે (પૂર્ણાંક j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_tq1_0_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_tq1_0 * GGML_RESTRICT x = vx;
    અચલ block_q8_K  * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    અચલ uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    દશાંશ sumf = 0.0f;

    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        પૂર્ણાંક sum = 0;

        માટે (size_t j = 0; j < કદ(x->qs) - કદ(x->qs) % 32; j += 32) {
            માટે (size_t l = 0; l < 5; ++l) {
                માટે (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        માટે (size_t j = કદ(x->qs) - કદ(x->qs) % 32; j < કદ(x->qs); j += 16) {
            માટે (size_t l = 0; l < 5; ++l) {
                માટે (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        માટે (size_t l = 0; l < 4; ++l) {
            માટે (size_t j = 0; j < કદ(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[કદ(x->qs)*5 + l*કદ(x->qh) + j];
            }
        }

        sumf += (દશાંશ) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_tq2_0_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_tq2_0 * GGML_RESTRICT x = vx;
    અચલ block_q8_K  * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;
    દશાંશ sumf = 0.0f;

    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        માટે (size_t j = 0; j < કદ(x->qs); j += 32) {
            માટે (size_t l = 0; l < 4; ++l) {
                માટે (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        અચલ દશાંશ d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumf += (દશાંશ) sumi * d;
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_q2_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q2_K * GGML_RESTRICT x = vx;
    અચલ block_q8_K * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    દશાંશ sumf = 0;

    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {

        અચલ uint8_t * q2 = x[i].qs;
        અચલ  int8_t * q8 = y[i].qs;
        અચલ uint8_t * sc = x[i].scales;

        પૂર્ણાંક summs = 0;
        માટે (પૂર્ણાંક j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        અચલ દશાંશ dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        અચલ દશાંશ dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        પૂર્ણાંક isum = 0;
        પૂર્ણાંક is = 0;
        પૂર્ણાંક d;
        માટે (પૂર્ણાંક k = 0; k < QK_K/128; ++k) {
            પૂર્ણાંક shift = 0;
            માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                પૂર્ણાંક isuml = 0;
                માટે (પૂર્ણાંક l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                માટે (પૂર્ણાંક l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

શૂન્ય ggml_vec_dot_q3_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ uint32_t kmask1 = 0x03030303;
    અચલ uint32_t kmask2 = 0x0f0f0f0f;

    અચલ block_q3_K * GGML_RESTRICT x = vx;
    અચલ block_q8_K * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    દશાંશ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*કદ(દશાંશ));

    uint32_t auxs[4];
    અચલ int8_t * scales = (અચલ int8_t*)auxs;

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        અચલ uint8_t * GGML_RESTRICT hm = x[i].hmask;
        અચલ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*કદ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        માટે (પૂર્ણાંક j = 0; j < QK_K; j += 128) {
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        માટે (પૂર્ણાંક j = 0; j < QK_K/16; ++j) {
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        માટે (પૂર્ણાંક l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    માટે (પૂર્ણાંક l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

શૂન્ય ggml_vec_dot_q4_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q4_K * GGML_RESTRICT x = vx;
    અચલ block_q8_K * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    સ્થિર અચલ uint32_t kmask1 = 0x3f3f3f3f;
    સ્થિર અચલ uint32_t kmask2 = 0x0f0f0f0f;
    સ્થિર અચલ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    અચલ uint8_t * scales = (અચલ uint8_t*)&utmp[0];
    અચલ uint8_t * mins   = (અચલ uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    દશાંશ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*કદ(દશાંશ));

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        અચલ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*કદ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        માટે (પૂર્ણાંક j = 0; j < QK_K/64; ++j) {
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        અચલ uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        પૂર્ણાંક sumi = 0;
        માટે (પૂર્ણાંક j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        પૂર્ણાંક is = 0;
        માટે (પૂર્ણાંક j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        માટે (પૂર્ણાંક l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        અચલ દશાંશ dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    માટે (પૂર્ણાંક l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

શૂન્ય ggml_vec_dot_q5_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy,  size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q5_K * GGML_RESTRICT x = vx;
    અચલ block_q8_K * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    સ્થિર અચલ uint32_t kmask1 = 0x3f3f3f3f;
    સ્થિર અચલ uint32_t kmask2 = 0x0f0f0f0f;
    સ્થિર અચલ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    અચલ uint8_t * scales = (અચલ uint8_t*)&utmp[0];
    અચલ uint8_t * mins   = (અચલ uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    દશાંશ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*કદ(દશાંશ));

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        અચલ uint8_t * GGML_RESTRICT hm = x[i].qh;
        અચલ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*કદ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        માટે (પૂર્ણાંક j = 0; j < QK_K/64; ++j) {
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        અચલ uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        પૂર્ણાંક sumi = 0;
        માટે (પૂર્ણાંક j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        પૂર્ણાંક is = 0;
        માટે (પૂર્ણાંક j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        માટે (પૂર્ણાંક l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        અચલ દશાંશ dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    માટે (પૂર્ણાંક l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

શૂન્ય ggml_vec_dot_q6_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_q6_K * GGML_RESTRICT x = vx;
    અચલ block_q8_K * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    દશાંશ   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*કદ(દશાંશ));

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ uint8_t * GGML_RESTRICT q4 = x[i].ql;
        અચલ uint8_t * GGML_RESTRICT qh = x[i].qh;
        અચલ  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*કદ(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        માટે (પૂર્ણાંક j = 0; j < QK_K; j += 128) {
            માટે (પૂર્ણાંક l = 0; l < 32; ++l) {
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
        પૂર્ણાંક is = 0;
        માટે (પૂર્ણાંક j = 0; j < QK_K/16; ++j) {
            પૂર્ણાંક scale = x[i].scales[is++];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            માટે (પૂર્ણાંક l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        માટે (પૂર્ણાંક l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    માટે (પૂર્ણાંક l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

શૂન્ય ggml_vec_dot_iq2_xxs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_iq2_xxs * GGML_RESTRICT x = vx;
    અચલ block_q8_K    * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    uint32_t aux32[2];
    અચલ uint8_t * aux8 = (અચલ uint8_t *)aux32;

    દશાંશ sumf = 0.f;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        અચલ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        અચલ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*કદ(uint32_t));
            q2 += 4;
            અચલ uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
                અચલ uint8_t * grid = (અચલ uint8_t *)(iq2xxs_grid + aux8[l]);
                અચલ uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
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

શૂન્ય ggml_vec_dot_iq2_xs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_iq2_xs * GGML_RESTRICT x = vx;
    અચલ block_q8_K   * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    દશાંશ sumf = 0.f;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        અચલ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        અચલ uint8_t  * GGML_RESTRICT sc = x[i].scales;
        અચલ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K/32; ++ib32) {
            અચલ uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            અચલ uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            માટે (પૂર્ણાંક l = 0; l < 2; ++l) {
                અચલ uint8_t * grid = (અચલ uint8_t *)(iq2xs_grid + (q2[l] & 511));
                અચલ uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            માટે (પૂર્ણાંક l = 2; l < 4; ++l) {
                અચલ uint8_t * grid = (અચલ uint8_t *)(iq2xs_grid + (q2[l] & 511));
                અચલ uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
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

શૂન્ય ggml_vec_dot_iq2_s_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_iq2_s * GGML_RESTRICT x = vx;
    અચલ block_q8_K  * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક i = 0; i < nb; i++) {

        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        અચલ int8_t  * q8 = y[i].qs;
        અચલ uint8_t * qs = x[i].qs;
        અચલ uint8_t * qh = x[i].qh;
        અચલ uint8_t * signs = qs + QK_K/8;

        પૂર્ણાંક bsum = 0;
        માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K/32; ++ib32) {
            પૂર્ણાંક ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            પૂર્ણાંક ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            પૂર્ણાંક sumi1 = 0, sumi2 = 0;
            માટે (પૂર્ણાંક l = 0; l < 2; ++l) {
                અચલ uint8_t * grid = (અચલ uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            માટે (પૂર્ણાંક l = 2; l < 4; ++l) {
                અચલ uint8_t * grid = (અચલ uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
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

શૂન્ય ggml_vec_dot_iq3_xxs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_iq3_xxs * GGML_RESTRICT x = vx;
    અચલ block_q8_K    * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    uint32_t aux32;

    દશાંશ sumf = 0.f;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        અચલ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        અચલ uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        અચલ int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, કદ(uint32_t)); gas += કદ(uint32_t);
            અચલ uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
                અચલ uint8_t * grid1 = (અચલ uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                અચલ uint8_t * grid2 = (અચલ uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                અચલ uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
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

શૂન્ય ggml_vec_dot_iq3_s_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_iq3_s * GGML_RESTRICT x = vx;
    અચલ block_q8_K  * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    દશાંશ sumf = 0.f;
    માટે (પૂર્ણાંક i = 0; i < nb; ++i) {
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        અચલ uint8_t * GGML_RESTRICT qs = x[i].qs;
        અચલ uint8_t * GGML_RESTRICT qh = x[i].qh;
        અચલ uint8_t * GGML_RESTRICT signs = x[i].signs;
        અચલ int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            અચલ uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            અચલ uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
                અચલ uint8_t * grid1 = (અચલ uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                અચલ uint8_t * grid2 = (અચલ uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
                અચલ uint8_t * grid1 = (અચલ uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                અચલ uint8_t * grid2 = (અચલ uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
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

શૂન્ય ggml_vec_dot_iq1_s_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_iq1_s * GGML_RESTRICT x = vx;
    અચલ block_q8_K  * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક i = 0; i < nb; i++) {

        અચલ int8_t   * q8 = y[i].qs;
        અચલ uint8_t  * qs = x[i].qs;
        અચલ uint16_t * qh = x[i].qh;

        પૂર્ણાંક sumi = 0, sumi1 = 0;
        માટે (પૂર્ણાંક ib = 0; ib < QK_K/32; ++ib) {
            અચલ પૂર્ણાંક ls = 2*((qh[ib] >> 12) & 7) + 1;
            અચલ પૂર્ણાંક delta = qh[ib] & 0x8000 ? -1 : 1;
            પૂર્ણાંક lsum = 0;
            માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
                અચલ int8_t * grid = (અચલ int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_iq1_m_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(n % QK_K == 0);
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    અચલ block_iq1_m * GGML_RESTRICT x = vx;
    અચલ block_q8_K  * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    iq1m_scale_t scale;

    પૂર્ણાંક sum1[2], sum2[2], delta[4];

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક i = 0; i < nb; i++) {

        અચલ int8_t   * q8 = y[i].qs;
        અચલ uint8_t  * qs = x[i].qs;
        અચલ uint8_t  * qh = x[i].qh;
        અચલ uint16_t * sc = (અચલ uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        પૂર્ણાંક sumi1 = 0, sumi2 = 0;
        માટે (પૂર્ણાંક ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
                અચલ int8_t * grid = (અચલ int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                પૂર્ણાંક lsum1 = 0, lsum2 = 0;
                માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            અચલ પૂર્ણાંક ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            અચલ પૂર્ણાંક ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

શૂન્ય ggml_vec_dot_iq4_nl_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    પ્રતિજ્ઞા(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    અચલ block_iq4_nl * GGML_RESTRICT x = vx;
    અચલ block_q8_0   * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK4_NL;

    પૂર્ણાંક ib = 0;
    દશાંશ sumf = 0;

    માટે (; ib < nb; ++ib) {
        અચલ દશાંશ d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        પૂર્ણાંક sumi1 = 0, sumi2 = 0;
        માટે (પૂર્ણાંક j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

શૂન્ય ggml_vec_dot_iq4_xs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc) {
    પ્રતિજ્ઞા(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    પ્રતિજ્ઞા(n % QK_K == 0);

    અચલ block_iq4_xs * GGML_RESTRICT x = vx;
    અચલ block_q8_K   * GGML_RESTRICT y = vy;

    અચલ પૂર્ણાંક nb = n / QK_K;

    દશાંશ sumf = 0;
    માટે (પૂર્ણાંક ibl = 0; ibl < nb; ++ibl) {
        અચલ દશાંશ d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        અચલ uint8_t * qs = x[ibl].qs;
        અચલ int8_t  * q8 = y[ibl].qs;
        માટે (પૂર્ણાંક ib = 0; ib < QK_K/32; ib += 2) {
            અચલ uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            અચલ uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            અચલ દશાંશ d1 = d4d8*(ls1 - 32);
            અચલ દશાંશ d2 = d4d8*(ls2 - 32);
            પૂર્ણાંક sumi1 = 0, sumi2 = 0;
            માટે (પૂર્ણાંક j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            માટે (પૂર્ણાંક j = 0; j < 16; ++j) {
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

શૂન્ય quantize_row_iq4_nl(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    પ્રતિજ્ઞા(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

શૂન્ય quantize_row_iq4_xs(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k) {
    પ્રતિજ્ઞા(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, ખાલી);
}
