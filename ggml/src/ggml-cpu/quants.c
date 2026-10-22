#define GGML_COMMON_IMPL_C
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "simd-mappings.h"
#अन्तर्भाव "ggml-quants.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव "arch-fallback.h"

#अन्तर्भाव <string.h>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <float.h>
#अन्तर्भाव <stdlib.h> // for qsort
#अन्तर्भाव <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

शून्य quantize_row_q1_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, y, k);
}

शून्य quantize_row_q2_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q2_0_ref(x, y, k);
}

शून्य quantize_row_q4_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

शून्य quantize_row_q4_1(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

शून्य quantize_row_q5_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

शून्य quantize_row_q5_1(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

शून्य quantize_row_q8_0_generic(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

शून्य quantize_row_q8_1_generic(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}

शून्य quantize_row_mxfp4(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_mxfp4_ref(x, y, k);
}

शून्य quantize_row_nvfp4(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_nvfp4_ref(x, y, k);
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

शून्य quantize_row_q2_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

शून्य quantize_row_q3_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

शून्य quantize_row_q4_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    प्रतिज्ञा(k % QK_K == 0);
    block_q4_K * GGML_RESTRICT y = vy;
    quantize_row_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

शून्य quantize_row_q5_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    प्रतिज्ञा(k % QK_K == 0);
    block_q5_K * GGML_RESTRICT y = vy;
    quantize_row_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

शून्य quantize_row_q6_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    प्रतिज्ञा(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

शून्य quantize_row_tq1_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    प्रतिज्ञा(k % QK_K == 0);
    block_tq1_0 * GGML_RESTRICT y = vy;
    quantize_row_tq1_0_ref(x, y, k);
}

शून्य quantize_row_tq2_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    प्रतिज्ञा(k % QK_K == 0);
    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

//===================================== Q8_K ==============================================

शून्य quantize_row_q8_K_generic(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

शून्य ggml_vec_dot_q1_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिरांक पूर्णांक qk = QK1_0;
    स्थिरांक पूर्णांक nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q1_0 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0 * GGML_RESTRICT y = vy;

    दशांश sumf = 0.0;

    साठी (पूर्णांक i = 0; i < nb; i++) {
        स्थिरांक दशांश d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        दशांश sumi = 0.0f;

        साठी (पूर्णांक k = 0; k < 4; k++) {
            स्थिरांक block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            स्थिरांक दशांश d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            पूर्णांक sumi_block = 0;

            स्थिरांक uint8_t * GGML_RESTRICT bits = &x[i].qs[k * 4];
            स्थिरांक int8_t  * GGML_RESTRICT qy   = yb->qs;

            साठी (पूर्णांक b = 0; b < 4; ++b, qy += 8) {
                स्थिरांक अचिन्हित mask = bits[b];
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

शून्य ggml_vec_dot_q2_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिरांक पूर्णांक qk = QK2_0;
    स्थिरांक पूर्णांक nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q2_0 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0 * GGML_RESTRICT y = vy;

    दशांश sumf = 0.0f;

    साठी (पूर्णांक i = 0; i < nb; i++) {
        स्थिरांक दशांश d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        दशांश sumi = 0.0f;

        // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
        साठी (पूर्णांक k = 0; k < 2; k++) {
            स्थिरांक block_q8_0 * GGML_RESTRICT yb = &y[i * 2 + k];
            स्थिरांक दशांश d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            पूर्णांक sumi_block = 0;

            स्थिरांक uint8_t * GGML_RESTRICT qs = &x[i].qs[k * 8];
            स्थिरांक int8_t  * GGML_RESTRICT qy = yb->qs;

            साठी (पूर्णांक b = 0; b < 8; ++b) {
                स्थिरांक uint8_t byte = qs[b];
                // Extract 4 two-bit values, map {0,1,2,3} -> {-1,0,1,2}
                sumi_block += ((पूर्णांक)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
                sumi_block += ((पूर्णांक)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
                sumi_block += ((पूर्णांक)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
                sumi_block += ((पूर्णांक)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

शून्य ggml_vec_dot_q4_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिरांक पूर्णांक qk = QK8_0;
    स्थिरांक पूर्णांक nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q4_0 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0 * GGML_RESTRICT y = vy;

    पूर्णांक ib = 0;
    दशांश sumf = 0;

    साठी (; ib < nb; ++ib) {
        पूर्णांक sumi0 = 0;
        पूर्णांक sumi1 = 0;

        साठी (पूर्णांक j = 0; j < qk/2; ++j) {
            स्थिरांक पूर्णांक v0 = (x[ib].qs[j] & 0x0F) - 8;
            स्थिरांक पूर्णांक v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        पूर्णांक sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
शून्य ggml_vec_dot_q4_1_q8_1_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिरांक पूर्णांक qk = QK8_1;
    स्थिरांक पूर्णांक nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q4_1 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_1 * GGML_RESTRICT y = vy;

    पूर्णांक ib = 0;
    दशांश sumf = 0;

    साठी (; ib < nb; ++ib) {
        पूर्णांक sumi0 = 0;
        पूर्णांक sumi1 = 0;

        साठी (पूर्णांक j = 0; j < qk/2; ++j) {
            स्थिरांक पूर्णांक v0 = (x[ib].qs[j] & 0x0F);
            स्थिरांक पूर्णांक v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        पूर्णांक sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

शून्य ggml_vec_dot_mxfp4_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    प्रतिज्ञा(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    स्थिरांक block_mxfp4 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0 * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_MXFP4;

    पूर्णांक ib = 0;
    दशांश sumf = 0;

    साठी (; ib < nb; ++ib) {
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);

        पूर्णांक sumi1 = 0;
        पूर्णांक sumi2 = 0;
        साठी (पूर्णांक j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
शून्य ggml_vec_dot_nvfp4_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    प्रतिज्ञा(n % QK_NVFP4 == 0);

    स्थिरांक block_nvfp4 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0 * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_NVFP4;

    दशांश sumf = 0;

    साठी (पूर्णांक ib = 0; ib < nb; ++ib) {
        साठी (पूर्णांक s_idx = 0; s_idx < 4; ++s_idx) {
            स्थिरांक दशांश d = ggml_ue4m3_to_fp32(x[ib].d[s_idx]);
            स्थिरांक पूर्णांक q8_block = s_idx / 2;
            स्थिरांक पूर्णांक q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            स्थिरांक दशांश dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            पूर्णांक sumi_lo = 0, sumi_hi = 0;
            साठी (पूर्णांक j = 0; j < QK_NVFP4_SUB/2; ++j) {
                स्थिरांक uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

शून्य ggml_vec_dot_q5_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिरांक पूर्णांक qk = QK8_0;
    स्थिरांक पूर्णांक nb = n / qk;

    पूर्णांक ib = 0;
    दशांश sumf = 0;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(qk == QK5_0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q5_0 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0 * GGML_RESTRICT y = vy;

    साठी (; ib < nb; ++ib) {
        uint32_t qh;
        स_नकल(&qh, x[ib].qh, आकार(qh));

        पूर्णांक sumi0 = 0;
        पूर्णांक sumi1 = 0;

        साठी (पूर्णांक j = 0; j < qk/2; ++j) {
            स्थिरांक uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            स्थिरांक uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            स्थिरांक int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            स्थिरांक int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        पूर्णांक sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

शून्य ggml_vec_dot_q5_1_q8_1_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिरांक पूर्णांक qk = QK8_1;
    स्थिरांक पूर्णांक nb = n / qk;

    पूर्णांक ib = 0;
    दशांश sumf = 0;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(qk == QK5_1);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q5_1 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_1 * GGML_RESTRICT y = vy;

    साठी (; ib < nb; ++ib) {
        uint32_t qh;
        स_नकल(&qh, x[ib].qh, आकार(qh));

        पूर्णांक sumi0 = 0;
        पूर्णांक sumi1 = 0;

        साठी (पूर्णांक j = 0; j < qk/2; ++j) {
            स्थिरांक uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            स्थिरांक uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            स्थिरांक int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            स्थिरांक int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        पूर्णांक sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

शून्य ggml_vec_dot_q8_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिरांक पूर्णांक qk = QK8_0;
    स्थिरांक पूर्णांक nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q8_0 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0 * GGML_RESTRICT y = vy;

    पूर्णांक ib = 0;
    दशांश sumf = 0;

    साठी (; ib < nb; ++ib) {
        पूर्णांक sumi = 0;

        साठी (पूर्णांक j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

शून्य ggml_vec_dot_tq1_0_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_tq1_0 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K  * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    स्थिरांक uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    दशांश sumf = 0.0f;

    साठी (पूर्णांक i = 0; i < nb; ++i) {
        पूर्णांक sum = 0;

        साठी (माप_प्रकार j = 0; j < आकार(x->qs) - आकार(x->qs) % 32; j += 32) {
            साठी (माप_प्रकार l = 0; l < 5; ++l) {
                साठी (माप_प्रकार m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        साठी (माप_प्रकार j = आकार(x->qs) - आकार(x->qs) % 32; j < आकार(x->qs); j += 16) {
            साठी (माप_प्रकार l = 0; l < 5; ++l) {
                साठी (माप_प्रकार m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        साठी (माप_प्रकार l = 0; l < 4; ++l) {
            साठी (माप_प्रकार j = 0; j < आकार(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[आकार(x->qs)*5 + l*आकार(x->qh) + j];
            }
        }

        sumf += (दशांश) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

शून्य ggml_vec_dot_tq2_0_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_tq2_0 * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K  * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;
    दशांश sumf = 0.0f;

    साठी (पूर्णांक i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        साठी (माप_प्रकार j = 0; j < आकार(x->qs); j += 32) {
            साठी (माप_प्रकार l = 0; l < 4; ++l) {
                साठी (माप_प्रकार k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        स्थिरांक दशांश d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumf += (दशांश) sumi * d;
    }

    *s = sumf;
}

शून्य ggml_vec_dot_q2_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q2_K * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    दशांश sumf = 0;

    साठी (पूर्णांक i = 0; i < nb; ++i) {

        स्थिरांक uint8_t * q2 = x[i].qs;
        स्थिरांक  int8_t * q8 = y[i].qs;
        स्थिरांक uint8_t * sc = x[i].scales;

        पूर्णांक summs = 0;
        साठी (पूर्णांक j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        स्थिरांक दशांश dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        स्थिरांक दशांश dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        पूर्णांक isum = 0;
        पूर्णांक is = 0;
        पूर्णांक d;
        साठी (पूर्णांक k = 0; k < QK_K/128; ++k) {
            पूर्णांक shift = 0;
            साठी (पूर्णांक j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                पूर्णांक isuml = 0;
                साठी (पूर्णांक l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                साठी (पूर्णांक l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
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

शून्य ggml_vec_dot_q3_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक uint32_t kmask1 = 0x03030303;
    स्थिरांक uint32_t kmask2 = 0x0f0f0f0f;

    स्थिरांक block_q3_K * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    दशांश   sums [8];
    int32_t aux32[8];
    स_रखो(sums, 0, 8*आकार(दशांश));

    uint32_t auxs[4];
    स्थिरांक int8_t * scales = (स्थिरांक int8_t*)auxs;

    दशांश sumf = 0;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक uint8_t * GGML_RESTRICT q3 = x[i].qs;
        स्थिरांक uint8_t * GGML_RESTRICT hm = x[i].hmask;
        स्थिरांक  int8_t * GGML_RESTRICT q8 = y[i].qs;
        स_रखो(aux32, 0, 8*आकार(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        साठी (पूर्णांक j = 0; j < QK_K; j += 128) {
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        स_नकल(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        साठी (पूर्णांक j = 0; j < QK_K/16; ++j) {
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        साठी (पूर्णांक l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    साठी (पूर्णांक l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

शून्य ggml_vec_dot_q4_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q4_K * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    स्थिर स्थिरांक uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर स्थिरांक uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर स्थिरांक uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    स्थिरांक uint8_t * scales = (स्थिरांक uint8_t*)&utmp[0];
    स्थिरांक uint8_t * mins   = (स्थिरांक uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    दशांश   sums [8];
    int32_t aux32[8];
    स_रखो(sums, 0, 8*आकार(दशांश));

    दशांश sumf = 0;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक uint8_t * GGML_RESTRICT q4 = x[i].qs;
        स्थिरांक  int8_t * GGML_RESTRICT q8 = y[i].qs;
        स_रखो(aux32, 0, 8*आकार(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        साठी (पूर्णांक j = 0; j < QK_K/64; ++j) {
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        स_नकल(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        स्थिरांक uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        पूर्णांक sumi = 0;
        साठी (पूर्णांक j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        पूर्णांक is = 0;
        साठी (पूर्णांक j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        साठी (पूर्णांक l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        स्थिरांक दशांश dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    साठी (पूर्णांक l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

शून्य ggml_vec_dot_q5_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy,  माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q5_K * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    स्थिर स्थिरांक uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर स्थिरांक uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर स्थिरांक uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    स्थिरांक uint8_t * scales = (स्थिरांक uint8_t*)&utmp[0];
    स्थिरांक uint8_t * mins   = (स्थिरांक uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    दशांश   sums [8];
    int32_t aux32[8];
    स_रखो(sums, 0, 8*आकार(दशांश));

    दशांश sumf = 0;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक uint8_t * GGML_RESTRICT q4 = x[i].qs;
        स्थिरांक uint8_t * GGML_RESTRICT hm = x[i].qh;
        स्थिरांक  int8_t * GGML_RESTRICT q8 = y[i].qs;
        स_रखो(aux32, 0, 8*आकार(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        साठी (पूर्णांक j = 0; j < QK_K/64; ++j) {
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            साठी (पूर्णांक l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        स_नकल(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        स्थिरांक uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        पूर्णांक sumi = 0;
        साठी (पूर्णांक j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        पूर्णांक is = 0;
        साठी (पूर्णांक j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        साठी (पूर्णांक l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        स्थिरांक दशांश dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    साठी (पूर्णांक l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

शून्य ggml_vec_dot_q6_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_q6_K * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    दशांश   sums [8];
    int32_t aux32[8];
    स_रखो(sums, 0, 8*आकार(दशांश));

    दशांश sumf = 0;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक uint8_t * GGML_RESTRICT q4 = x[i].ql;
        स्थिरांक uint8_t * GGML_RESTRICT qh = x[i].qh;
        स्थिरांक  int8_t * GGML_RESTRICT q8 = y[i].qs;
        स_रखो(aux32, 0, 8*आकार(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        साठी (पूर्णांक j = 0; j < QK_K; j += 128) {
            साठी (पूर्णांक l = 0; l < 32; ++l) {
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
        पूर्णांक is = 0;
        साठी (पूर्णांक j = 0; j < QK_K/16; ++j) {
            पूर्णांक scale = x[i].scales[is++];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            साठी (पूर्णांक l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            साठी (पूर्णांक l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        साठी (पूर्णांक l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    साठी (पूर्णांक l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

शून्य ggml_vec_dot_iq2_xxs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_iq2_xxs * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K    * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    uint32_t aux32[2];
    स्थिरांक uint8_t * aux8 = (स्थिरांक uint8_t *)aux32;

    दशांश sumf = 0.f;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        स्थिरांक uint16_t * GGML_RESTRICT q2 = x[i].qs;
        स्थिरांक int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        साठी (पूर्णांक ib32 = 0; ib32 < QK_K/32; ++ib32) {
            स_नकल(aux32, q2, 2*आकार(uint32_t));
            q2 += 4;
            स्थिरांक uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            साठी (पूर्णांक l = 0; l < 4; ++l) {
                स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *)(iq2xxs_grid + aux8[l]);
                स्थिरांक uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                साठी (पूर्णांक j = 0; j < 8; ++j) {
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

शून्य ggml_vec_dot_iq2_xs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_iq2_xs * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K   * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    दशांश sumf = 0.f;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        स्थिरांक uint16_t * GGML_RESTRICT q2 = x[i].qs;
        स्थिरांक uint8_t  * GGML_RESTRICT sc = x[i].scales;
        स्थिरांक int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        साठी (पूर्णांक ib32 = 0; ib32 < QK_K/32; ++ib32) {
            स्थिरांक uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            स्थिरांक uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            साठी (पूर्णांक l = 0; l < 2; ++l) {
                स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *)(iq2xs_grid + (q2[l] & 511));
                स्थिरांक uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                साठी (पूर्णांक j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            साठी (पूर्णांक l = 2; l < 4; ++l) {
                स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *)(iq2xs_grid + (q2[l] & 511));
                स्थिरांक uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                साठी (पूर्णांक j = 0; j < 8; ++j) {
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

शून्य ggml_vec_dot_iq2_s_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_iq2_s * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K  * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    दशांश sumf = 0;
    साठी (पूर्णांक i = 0; i < nb; i++) {

        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        स्थिरांक int8_t  * q8 = y[i].qs;
        स्थिरांक uint8_t * qs = x[i].qs;
        स्थिरांक uint8_t * qh = x[i].qh;
        स्थिरांक uint8_t * signs = qs + QK_K/8;

        पूर्णांक bsum = 0;
        साठी (पूर्णांक ib32 = 0; ib32 < QK_K/32; ++ib32) {
            पूर्णांक ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            पूर्णांक ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            पूर्णांक sumi1 = 0, sumi2 = 0;
            साठी (पूर्णांक l = 0; l < 2; ++l) {
                स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                साठी (पूर्णांक j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            साठी (पूर्णांक l = 2; l < 4; ++l) {
                स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                साठी (पूर्णांक j = 0; j < 8; ++j) {
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

शून्य ggml_vec_dot_iq3_xxs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_iq3_xxs * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K    * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    uint32_t aux32;

    दशांश sumf = 0.f;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        स्थिरांक uint8_t * GGML_RESTRICT q3 = x[i].qs;
        स्थिरांक uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        स्थिरांक int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        साठी (पूर्णांक ib32 = 0; ib32 < QK_K/32; ++ib32) {
            स_नकल(&aux32, gas, आकार(uint32_t)); gas += आकार(uint32_t);
            स्थिरांक uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            साठी (पूर्णांक l = 0; l < 4; ++l) {
                स्थिरांक uint8_t * grid1 = (स्थिरांक uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                स्थिरांक uint8_t * grid2 = (स्थिरांक uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                स्थिरांक uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                साठी (पूर्णांक j = 0; j < 4; ++j) {
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

शून्य ggml_vec_dot_iq3_s_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_iq3_s * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K  * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    दशांश sumf = 0.f;
    साठी (पूर्णांक i = 0; i < nb; ++i) {
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        स्थिरांक uint8_t * GGML_RESTRICT qs = x[i].qs;
        स्थिरांक uint8_t * GGML_RESTRICT qh = x[i].qh;
        स्थिरांक uint8_t * GGML_RESTRICT signs = x[i].signs;
        स्थिरांक int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        साठी (पूर्णांक ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            स्थिरांक uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            स्थिरांक uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            साठी (पूर्णांक l = 0; l < 4; ++l) {
                स्थिरांक uint8_t * grid1 = (स्थिरांक uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                स्थिरांक uint8_t * grid2 = (स्थिरांक uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                साठी (पूर्णांक j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            साठी (पूर्णांक l = 0; l < 4; ++l) {
                स्थिरांक uint8_t * grid1 = (स्थिरांक uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                स्थिरांक uint8_t * grid2 = (स्थिरांक uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                साठी (पूर्णांक j = 0; j < 4; ++j) {
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

शून्य ggml_vec_dot_iq1_s_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_iq1_s * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K  * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    दशांश sumf = 0;
    साठी (पूर्णांक i = 0; i < nb; i++) {

        स्थिरांक int8_t   * q8 = y[i].qs;
        स्थिरांक uint8_t  * qs = x[i].qs;
        स्थिरांक uint16_t * qh = x[i].qh;

        पूर्णांक sumi = 0, sumi1 = 0;
        साठी (पूर्णांक ib = 0; ib < QK_K/32; ++ib) {
            स्थिरांक पूर्णांक ls = 2*((qh[ib] >> 12) & 7) + 1;
            स्थिरांक पूर्णांक delta = qh[ib] & 0x8000 ? -1 : 1;
            पूर्णांक lsum = 0;
            साठी (पूर्णांक l = 0; l < 4; ++l) {
                स्थिरांक int8_t * grid = (स्थिरांक int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                साठी (पूर्णांक j = 0; j < 8; ++j) {
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

शून्य ggml_vec_dot_iq1_m_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिरांक block_iq1_m * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K  * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    iq1m_scale_t scale;

    पूर्णांक sum1[2], sum2[2], delta[4];

    दशांश sumf = 0;
    साठी (पूर्णांक i = 0; i < nb; i++) {

        स्थिरांक int8_t   * q8 = y[i].qs;
        स्थिरांक uint8_t  * qs = x[i].qs;
        स्थिरांक uint8_t  * qh = x[i].qh;
        स्थिरांक uint16_t * sc = (स्थिरांक uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        पूर्णांक sumi1 = 0, sumi2 = 0;
        साठी (पूर्णांक ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            साठी (पूर्णांक l = 0; l < 4; ++l) {
                स्थिरांक int8_t * grid = (स्थिरांक int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                पूर्णांक lsum1 = 0, lsum2 = 0;
                साठी (पूर्णांक j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            स्थिरांक पूर्णांक ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            स्थिरांक पूर्णांक ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

शून्य ggml_vec_dot_iq4_nl_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    प्रतिज्ञा(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    स्थिरांक block_iq4_nl * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_0   * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK4_NL;

    पूर्णांक ib = 0;
    दशांश sumf = 0;

    साठी (; ib < nb; ++ib) {
        स्थिरांक दशांश d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        पूर्णांक sumi1 = 0, sumi2 = 0;
        साठी (पूर्णांक j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

शून्य ggml_vec_dot_iq4_xs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    प्रतिज्ञा(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    प्रतिज्ञा(n % QK_K == 0);

    स्थिरांक block_iq4_xs * GGML_RESTRICT x = vx;
    स्थिरांक block_q8_K   * GGML_RESTRICT y = vy;

    स्थिरांक पूर्णांक nb = n / QK_K;

    दशांश sumf = 0;
    साठी (पूर्णांक ibl = 0; ibl < nb; ++ibl) {
        स्थिरांक दशांश d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        स्थिरांक uint8_t * qs = x[ibl].qs;
        स्थिरांक int8_t  * q8 = y[ibl].qs;
        साठी (पूर्णांक ib = 0; ib < QK_K/32; ib += 2) {
            स्थिरांक uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            स्थिरांक uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            स्थिरांक दशांश d1 = d4d8*(ls1 - 32);
            स्थिरांक दशांश d2 = d4d8*(ls2 - 32);
            पूर्णांक sumi1 = 0, sumi2 = 0;
            साठी (पूर्णांक j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            साठी (पूर्णांक j = 0; j < 16; ++j) {
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

शून्य quantize_row_iq4_nl(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    प्रतिज्ञा(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

शून्य quantize_row_iq4_xs(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k) {
    प्रतिज्ञा(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, रिक्त);
}
