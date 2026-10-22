#define GGML_COMMON_IMPL_C
#অন্তর্ভাৱ "ggml-common.h"

#অন্তর্ভাৱ "ggml-quants.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-cpu/ggml-cpu-impl.h"
#অন্তর্ভাৱ "ggml-cpu.h"

#অন্তর্ভাৱ <math.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <assert.h>
#অন্তর্ভাৱ <float.h>
#অন্তর্ভাৱ <stdlib.h> // for qsort
#অন্তর্ভাৱ <stdio.h>  // for GGML_ASSERT

#ifdef GGML_USE_OPENMP
#অন্তর্ভাৱ <omp.h>
#endif

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

স্থির অন্তঃস্থ পূর্ণাঙ্ক best_index_int8(পূর্ণাঙ্ক n, ধ্রুৱ int8_t * val, ভিন্ন x) {
    যদি (x <= val[0]) ঘূৰাই 0;
    যদি (x >= val[n-1]) ঘূৰাই n-1;
    পূর্ণাঙ্ক ml = 0, mu = n-1;
    যেতিয়ালৈ (mu-ml > 1) {
        পূর্ণাঙ্ক mav = (ml+mu)/2;
        যদি (x < val[mav]) mu = mav; অন্যথা ml = mav;
    }
    ঘূৰাই x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// reference implementation for deterministic creation of model files
শূন্য quantize_row_q1_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK1_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন sum_abs = 0.0f;
        বাবে (পূর্ণাঙ্ক j = 0; j < qk; j++) {
            sum_abs += fabsf(x[i*qk + j]);
        }
        ধ্রুৱ ভিন্ন d = sum_abs / qk;

        y[i].d = GGML_FP32_TO_FP16(d);

        // Clear all bits first
        বাবে (পূর্ণাঙ্ক j = 0; j < qk / 8; ++j) {
            y[i].qs[j] = 0;
        }

        // Just store sign of each weight directly (no normalization)
        বাবে (পূর্ণাঙ্ক j = 0; j < qk; ++j) {
            ধ্রুৱ পূর্ণাঙ্ক bit_index = j;
            ধ্রুৱ পূর্ণাঙ্ক byte_index = bit_index / 8;
            ধ্রুৱ পূর্ণাঙ্ক bit_offset = bit_index % 8;

            যদি (x[i*qk + j] >= 0.0f) {
                y[i].qs[byte_index] |= (1 << bit_offset);
            }
        }
    }
}

শূন্য quantize_row_q2_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK2_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        // Compute scale as max absolute value in the block
        ভিন্ন amax = 0.0f;
        বাবে (পূর্ণাঙ্ক j = 0; j < qk; j++) {
            ধ্রুৱ ভিন্ন a = fabsf(x[i*qk + j]);
            যদি (a > amax) amax = a;
        }
        ধ্রুৱ ভিন্ন d = amax;
        ধ্রুৱ ভিন্ন id = d > 0.0f ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // Clear quant bytes
        বাবে (পূর্ণাঙ্ক j = 0; j < qk / 4; ++j) {
            y[i].qs[j] = 0;
        }

        // Encode 2-bit values: round(w/d) clamped to [-1, 2], then add 1
        // 00 (-1) = -scale, 01 (0) = 0, 10 (+1) = +scale, 11 (+2) = 2*scale
        বাবে (পূর্ণাঙ্ক j = 0; j < qk; ++j) {
            ধ্রুৱ ভিন্ন w = x[i*qk + j];
            পূর্ণাঙ্ক q = (পূর্ণাঙ্ক)roundf(w * id) + 1;
            যদি (q < 0) q = 0;
            যদি (q > 3) q = 3;
            ধ্রুৱ পূর্ণাঙ্ক byte_index = j / 4;
            ধ্রুৱ পূর্ণাঙ্ক bit_offset = (j % 4) * 2;
            y[i].qs[byte_index] |= ((uint8_t)q << bit_offset);
        }
    }
}

// reference implementation for deterministic creation of model files
শূন্য quantize_row_q4_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK4_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন amax = 0.0f; // absolute max
        ভিন্ন max  = 0.0f;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; j++) {
            ধ্রুৱ ভিন্ন v = x[i*qk + j];
            যদি (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        ধ্রুৱ ভিন্ন d  = max / -8;
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ ভিন্ন x0 = x[i*qk + 0    + j]*id;
            ধ্রুৱ ভিন্ন x1 = x[i*qk + qk/2 + j]*id;

            ধ্রুৱ uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            ধ্রুৱ uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

শূন্য quantize_row_q4_1_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k) {
    ধ্রুৱ পূর্ণাঙ্ক qk = QK4_1;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন min = FLT_MAX;
        ভিন্ন max = -FLT_MAX;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; j++) {
            ধ্রুৱ ভিন্ন v = x[i*qk + j];

            যদি (v < min) min = v;
            যদি (v > max) max = v;
        }

        ধ্রুৱ ভিন্ন d  = (max - min) / ((1 << 4) - 1);
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ ভিন্ন x0 = (x[i*qk + 0    + j] - min)*id;
            ধ্রুৱ ভিন্ন x1 = (x[i*qk + qk/2 + j] - min)*id;

            ধ্রুৱ uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            ধ্রুৱ uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

শূন্য quantize_row_q5_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK5_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন amax = 0.0f; // absolute max
        ভিন্ন max  = 0.0f;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; j++) {
            ধ্রুৱ ভিন্ন v = x[i*qk + j];
            যদি (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        ধ্রুৱ ভিন্ন d  = max / -16;
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ ভিন্ন x0 = x[i*qk + 0    + j]*id;
            ধ্রুৱ ভিন্ন x1 = x[i*qk + qk/2 + j]*id;

            ধ্রুৱ uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            ধ্রুৱ uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, পরিমাণ(qh));
    }
}

শূন্য quantize_row_q5_1_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k) {
    ধ্রুৱ পূর্ণাঙ্ক qk = QK5_1;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন min = FLT_MAX;
        ভিন্ন max = -FLT_MAX;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; j++) {
            ধ্রুৱ ভিন্ন v = x[i*qk + j];

            যদি (v < min) min = v;
            যদি (v > max) max = v;
        }

        ধ্রুৱ ভিন্ন d  = (max - min) / ((1 << 5) - 1);
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        uint32_t qh = 0;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ ভিন্ন x0 = (x[i*qk + 0    + j] - min)*id;
            ধ্রুৱ ভিন্ন x1 = (x[i*qk + qk/2 + j] - min)*id;

            ধ্রুৱ uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            ধ্রুৱ uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, পরিমাণ(y[i].qh));
    }
}

// reference implementation for deterministic creation of model files
শূন্য quantize_row_q8_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK8_0 == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK8_0;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন amax = 0.0f; // absolute max

        বাবে (পূর্ণাঙ্ক j = 0; j < QK8_0; j++) {
            ধ্রুৱ ভিন্ন v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        ধ্রুৱ ভিন্ন d = amax / ((1 << 7) - 1);
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        বাবে (পূর্ণাঙ্ক j = 0; j < QK8_0; ++j) {
            ধ্রুৱ ভিন্ন x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

// reference implementation for deterministic creation of model files
শূন্য quantize_row_q8_1_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(QK8_1 == 32);
    প্রতিজ্ঞা(k % QK8_1 == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK8_1;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন amax = 0.0f; // absolute max

        বাবে (পূর্ণাঙ্ক j = 0; j < QK8_1; j++) {
            ধ্রুৱ ভিন্ন v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        ধ্রুৱ ভিন্ন d = amax / ((1 << 7) - 1);
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        পূর্ণাঙ্ক sum = 0;

        বাবে (পূর্ণাঙ্ক j = 0; j < QK8_1/2; ++j) {
            ধ্রুৱ ভিন্ন v0 = x[i*QK8_1           + j]*id;
            ধ্রুৱ ভিন্ন v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        y[i].s = GGML_FP32_TO_FP16(sum*d);
    }
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক best_index_mxfp4(ভিন্ন x, ভিন্ন e) {
    পূর্ণাঙ্ক best_index = 0;
    ভিন্ন best_err = fabsf(kvalues_mxfp4[0]*e - x);
    বাবে (পূর্ণাঙ্ক i = 1; i < 16; i++) {
        ভিন্ন err = fabsf(kvalues_mxfp4[i]*e - x);
        যদি (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    ঘূৰাই best_index;
}

শূন্য quantize_row_mxfp4_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK_MXFP4;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন amax = 0.0f; // absolute max

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; j++) {
            ধ্রুৱ ভিন্ন v = x[i*qk + j];

            যদি (amax < fabsf(v)) {
                amax = fabsf(v);
            }
        }

        ধ্রুৱ uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;

        ধ্রুৱ ভিন্ন d = GGML_E8M0_TO_FP32_HALF(e);

        y[i].e = e;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ uint8_t x0 = best_index_mxfp4(x[i*qk + 0    + j], d);
            ধ্রুৱ uint8_t x1 = best_index_mxfp4(x[i*qk + qk/2 + j], d);

            y[i].qs[j]  = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

শূন্য quantize_row_nvfp4_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK_NVFP4;
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk_sub = QK_NVFP4_SUB;
    স্থির ধ্রুৱ পূর্ণাঙ্ক n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        বাবে (পূর্ণাঙ্ক s = 0; s < n_sub; s++) {
            ধ্রুৱ ভিন্ন * xb = x + i*qk + s*qk_sub;

            ভিন্ন amax = 0.0f;
            বাবে (পূর্ণাঙ্ক j = 0; j < qk_sub; j++) {
                যদি (amax < fabsf(xb[j])) {
                    amax = fabsf(xb[j]);
                }
            }

            // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
            ধ্রুৱ uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
            y[i].d[s] = ue;
            ধ্রুৱ ভিন্ন d = ggml_ue4m3_to_fp32(ue);

            বাবে (পূর্ণাঙ্ক j = 0; j < qk_sub/2; ++j) {
                ধ্রুৱ uint8_t x0 = best_index_mxfp4(xb[0        + j], d);
                ধ্রুৱ uint8_t x1 = best_index_mxfp4(xb[qk_sub/2 + j], d);

                y[i].qs[s*(qk_sub/2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

শূন্য dequantize_row_q1_0(ধ্রুৱ block_q1_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK1_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ ভিন্ন neg_d = -d;

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; ++j) {
            ধ্রুৱ পূর্ণাঙ্ক byte_index = j / 8;
            ধ্রুৱ পূর্ণাঙ্ক bit_offset = j % 8;
            ধ্রুৱ uint8_t bit = (x[i].qs[byte_index] >> bit_offset) & 1;
            y[i*qk + j] = bit ? d : neg_d;
        }
    }
}

শূন্য dequantize_row_q2_0(ধ্রুৱ block_q2_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK2_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; ++j) {
            ধ্রুৱ পূর্ণাঙ্ক byte_index = j / 4;
            ধ্রুৱ পূর্ণাঙ্ক bit_offset = (j % 4) * 2;
            ধ্রুৱ uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            // 00=-1, 01=0, 10=+1, 11=+2
            y[i*qk + j] = ((পূর্ণাঙ্ক)q - 1) * d;
        }
    }
}

শূন্য dequantize_row_q4_0(ধ্রুৱ block_q4_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK4_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ পূর্ণাঙ্ক x0 = (x[i].qs[j] & 0x0F) - 8;
            ধ্রুৱ পূর্ণাঙ্ক x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

শূন্য dequantize_row_q4_1(ধ্রুৱ block_q4_1 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK4_1;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ ভিন্ন m = GGML_FP16_TO_FP32(x[i].m);

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ পূর্ণাঙ্ক x0 = (x[i].qs[j] & 0x0F);
            ধ্রুৱ পূর্ণাঙ্ক x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

শূন্য dequantize_row_q5_0(ধ্রুৱ block_q5_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK5_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        uint32_t qh;
        memcpy(&qh, x[i].qh, পরিমাণ(qh));

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            ধ্রুৱ uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            ধ্রুৱ int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            ধ্রুৱ int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

শূন্য dequantize_row_q5_1(ধ্রুৱ block_q5_1 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK5_1;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ ভিন্ন m = GGML_FP16_TO_FP32(x[i].m);

        uint32_t qh;
        memcpy(&qh, x[i].qh, পরিমাণ(qh));

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            ধ্রুৱ uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            ধ্রুৱ পূর্ণাঙ্ক x0 = (x[i].qs[j] & 0x0F) | xh_0;
            ধ্রুৱ পূর্ণাঙ্ক x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

শূন্য dequantize_row_q8_0(ধ্রুৱ block_q8_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK8_0;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (পূর্ণাঙ্ক j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

শূন্য dequantize_row_mxfp4(ধ্রুৱ block_mxfp4 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK_MXFP4;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_E8M0_TO_FP32_HALF(x[i].e);

        বাবে (পূর্ণাঙ্ক j = 0; j < qk/2; ++j) {
            ধ্রুৱ int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            ধ্রুৱ int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

শূন্য dequantize_row_nvfp4(ধ্রুৱ block_nvfp4 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk = QK_NVFP4;
    স্থির ধ্রুৱ পূর্ণাঙ্ক qk_sub = QK_NVFP4_SUB;
    স্থির ধ্রুৱ পূর্ণাঙ্ক n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    প্রতিজ্ঞা(k % qk == 0);

    ধ্রুৱ পূর্ণাঙ্ক nb = k / qk;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        বাবে (পূর্ণাঙ্ক s = 0; s < n_sub; s++) {
            ধ্রুৱ ভিন্ন d = ggml_ue4m3_to_fp32(x[i].d[s]);
            ভিন্ন * yb = y + i*qk + s*qk_sub;

            বাবে (পূর্ণাঙ্ক j = 0; j < qk_sub/2; ++j) {
                ধ্রুৱ int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
                ধ্রুৱ int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];

                yb[j + 0       ] = v0*d;
                yb[j + qk_sub/2] = v1*d;
            }
        }
    }
}

//
// 2-6 bit quantization in super-blocks
//

//
// ===================== Helper functions
//
স্থির অন্তঃস্থ পূর্ণাঙ্ক nearest_int(ভিন্ন fval) {
    প্রতিজ্ঞা(fabsf(fval) <= 4194303.f);
    ভিন্ন val = fval + 12582912.f;
    পূর্ণাঙ্ক i; memcpy(&i, &val, পরিমাণ(পূর্ণাঙ্ক));
    ঘূৰাই (i & 0x007fffff) - 0x00400000;
}

স্থির ভিন্ন make_qx_quants(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, পূর্ণাঙ্ক rmse_type,
        ধ্রুৱ ভিন্ন * GGML_RESTRICT qw) {
    ভিন্ন max = 0;
    ভিন্ন amax = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        ভিন্ন ax = fabsf(x[i]);
        যদি (ax > amax) { amax = ax; max = x[i]; }
    }
    যদি (amax < GROUP_MAX_EPS) { // all zero
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            L[i] = 0;
        }
        ঘূৰাই 0.f;
    }
    ভিন্ন iscale = -nmax / max;
    যদি (rmse_type == 0) {
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        ঘূৰাই 1/iscale;
    }
    bool return_early = false;
    যদি (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    ভিন্ন sumlx = 0;
    ভিন্ন suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    বাবে (চঞ্চল পূর্ণাঙ্ক i = 0; i < n; ++i) {
#অন্যথা
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
#endif
        পূর্ণাঙ্ক l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        ভিন্ন w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    ভিন্ন scale = suml2 ? sumlx/suml2 : 0.0f;
    যদি (return_early) ঘূৰাই suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    ভিন্ন best = scale * sumlx;
    বাবে (পূর্ণাঙ্ক is = -9; is <= 9; ++is) {
        যদি (is == 0) {
            অনুৱর্ত;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            ভিন্ন w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        যদি (suml2 > 0 && sumlx*sumlx > best*suml2) {
            বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                পূর্ণাঙ্ক l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    ঘূৰাই scale;
}

স্থির ভিন্ন make_q3_quants(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, bool do_rmse) {
    ভিন্ন max = 0;
    ভিন্ন amax = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        ভিন্ন ax = fabsf(x[i]);
        যদি (ax > amax) { amax = ax; max = x[i]; }
    }
    যদি (amax < GROUP_MAX_EPS) { // all zero
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) { L[i] = 0; }
        ঘূৰাই 0.f;
    }
    ভিন্ন iscale = -nmax / max;
    যদি (do_rmse) {
        ভিন্ন sumlx = 0;
        ভিন্ন suml2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            ভিন্ন w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        বাবে (পূর্ণাঙ্ক itry = 0; itry < 5; ++itry) {
            পূর্ণাঙ্ক n_changed = 0;
            বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                ভিন্ন w = x[i]*x[i];
                ভিন্ন slx = sumlx - w*x[i]*L[i];
                যদি (slx > 0) {
                    ভিন্ন sl2 = suml2 - w*L[i]*L[i];
                    পূর্ণাঙ্ক new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    যদি (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        যদি (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            যদি (!n_changed) {
                ভঙ্গ;
            }
        }
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        ঘূৰাই suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        পূর্ণাঙ্ক l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    ঘূৰাই 1/iscale;
}

স্থির ভিন্ন make_qkx1_quants(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, ভিন্ন * GGML_RESTRICT the_min,
        পূর্ণাঙ্ক ntry, ভিন্ন alpha) {
    ভিন্ন min = x[0];
    ভিন্ন max = x[0];
    বাবে (পূর্ণাঙ্ক i = 1; i < n; ++i) {
        যদি (x[i] < min) min = x[i];
        যদি (x[i] > max) max = x[i];
    }
    যদি (max == min) {
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        ঘূৰাই 0.f;
    }
    যদি (min > 0) min = 0;
    ভিন্ন iscale = nmax/(max - min);
    ভিন্ন scale = 1/iscale;
    বাবে (পূর্ণাঙ্ক itry = 0; itry < ntry; ++itry) {
        ভিন্ন sumlx = 0; পূর্ণাঙ্ক suml2 = 0;
        bool did_change = false;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            যদি (l != L[i]) {
                L[i] = l;
                did_change = true;
            }
            sumlx += (x[i] - min)*l;
            suml2 += l*l;
        }
        scale = sumlx/suml2;
        ভিন্ন sum = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            sum += x[i] - scale*L[i];
        }
        min = alpha*min + (1 - alpha)*sum/n;
        যদি (min > 0) min = 0;
        iscale = 1/scale;
        যদি (!did_change) ভঙ্গ;
    }
    *the_min = -min;
    ঘূৰাই scale;
}

স্থির ভিন্ন make_qkx2_quants(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, ধ্রুৱ ভিন্ন * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, ভিন্ন * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        ভিন্ন rmin, ভিন্ন rdelta, পূর্ণাঙ্ক nstep, bool use_mad) {
    ভিন্ন min = x[0];
    ভিন্ন max = x[0];
    ভিন্ন sum_w = weights[0];
    ভিন্ন sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    বাবে (চঞ্চল পূর্ণাঙ্ক i = 1; i < n; ++i) {
#অন্যথা
    বাবে (পূর্ণাঙ্ক i = 1; i < n; ++i) {
#endif
        যদি (x[i] < min) min = x[i];
        যদি (x[i] > max) max = x[i];
        ভিন্ন w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    যদি (min > 0) min = 0;
    যদি (max == min) {
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        ঘূৰাই 0.f;
    }
    ভিন্ন iscale = nmax/(max - min);
    ভিন্ন scale = 1/iscale;
    ভিন্ন best_error = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        পূর্ণাঙ্ক l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        ভিন্ন diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        ভিন্ন w = weights[i];
        best_error += w * diff;
    }
    যদি (nstep < 1) {
        *the_min = -min;
        ঘূৰাই scale;
    }
    বাবে (পূর্ণাঙ্ক is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        ভিন্ন sum_l = 0, sum_l2 = 0, sum_xl = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            ভিন্ন w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        ভিন্ন D = sum_w * sum_l2 - sum_l * sum_l;
        যদি (D > 0) {
            ভিন্ন this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            ভিন্ন this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            যদি (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            ভিন্ন cur_error = 0;
            বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                ভিন্ন diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                ভিন্ন w = weights[i];
                cur_error += w * diff;
            }
            যদি (cur_error < best_error) {
                বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    ঘূৰাই scale;
}

স্থির অন্তঃস্থ শূন্য get_scale_min_k4(পূর্ণাঙ্ক j, ধ্রুৱ uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    যদি (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } অন্যথা {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

//========================- 2-bit (de)-quantization

শূন্য quantize_row_q2_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    ভিন্ন   weights[16];
    ভিন্ন mins[QK_K/16];
    ভিন্ন scales[QK_K/16];

    ধ্রুৱ ভিন্ন q4scale = 15.f;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন max_scale = 0; // as we are deducting the min, scales are always positive
        ভিন্ন max_min = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            ভিন্ন scale = scales[j];
            যদি (scale > max_scale) {
                max_scale = scale;
            }
            ভিন্ন min = mins[j];
            যদি (min > max_min) {
                max_min = min;
            }
        }

        যদি (max_scale > 0) {
            ভিন্ন iscale = q4scale/max_scale;
            বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
                পূর্ণাঙ্ক l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } অন্যথা {
            বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        যদি (max_min > 0) {
            ভিন্ন iscale = q4scale/max_min;
            বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
                পূর্ণাঙ্ক l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } অন্যথা {
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            যদি (!d) অনুৱর্ত;
            ধ্রুৱ ভিন্ন dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 16; ++ii) {
                পূর্ণাঙ্ক l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 128) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

শূন্য dequantize_row_q2_K(ধ্রুৱ block_q2_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ ভিন্ন min = GGML_FP16_TO_FP32(x[i].dmin);

        ধ্রুৱ uint8_t * q = x[i].qs;

        পূর্ণাঙ্ক is = 0;
        ভিন্ন dl, ml;
        বাবে (পূর্ণাঙ্ক n = 0; n < QK_K; n += 128) {
            পূর্ণাঙ্ক shift = 0;
            বাবে (পূর্ণাঙ্ক j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

স্থির ভিন্ন make_qkx3_quants(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, ধ্রুৱ ভিন্ন * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, ভিন্ন * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        ভিন্ন rmin, ভিন্ন rdelta, পূর্ণাঙ্ক nstep, bool use_mad) {
    ভিন্ন min = x[0];
    ভিন্ন max = x[0];
    ভিন্ন sum_w = weights ? weights[0] : x[0]*x[0];
    ভিন্ন sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    বাবে (চঞ্চল পূর্ণাঙ্ক i = 1; i < n; ++i) {
#অন্যথা
    বাবে (পূর্ণাঙ্ক i = 1; i < n; ++i) {
#endif
        যদি (x[i] < min) min = x[i];
        যদি (x[i] > max) max = x[i];
        ভিন্ন w = weights ? weights[i] : x[i]*x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    যদি (min > 0) {
        min = 0;
    }
    যদি (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        ঘূৰাই 0.f;
    }
    ভিন্ন iscale = nmax/(max - min);
    ভিন্ন scale = 1/iscale;
    ভিন্ন best_mad = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        পূর্ণাঙ্ক l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        ভিন্ন diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        ভিন্ন w = weights ? weights[i] : x[i]*x[i];
        best_mad += w * diff;
    }
    যদি (nstep < 1) {
        *the_min = -min;
        ঘূৰাই scale;
    }
    বাবে (পূর্ণাঙ্ক is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        ভিন্ন sum_l = 0, sum_l2 = 0, sum_xl = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            // min is the best fit so far and can be at or near max, so v can be inf, nan or out of range for nearest_int
            ধ্রুৱ ভিন্ন v = iscale*(x[i] - min);
            ধ্রুৱ পূর্ণাঙ্ক l = v > 0 ? nearest_int(MIN(v, nmax)) : 0;
            Laux[i] = l;
            ভিন্ন w = weights ? weights[i] : x[i]*x[i];
            sum_l  += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        ভিন্ন D = sum_w * sum_l2 - sum_l * sum_l;
        যদি (D > 0) {
            ভিন্ন this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            ভিন্ন this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            যদি (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            ভিন্ন mad = 0;
            বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                ভিন্ন diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                ভিন্ন w = weights ? weights[i] : x[i]*x[i];
                mad += w * diff;
            }
            যদি (mad < best_mad) {
                বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    ঘূৰাই scale;
}

স্থির ভিন্ন make_qp_quants(পূর্ণাঙ্ক n, পূর্ণাঙ্ক nmax, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, ধ্রুৱ ভিন্ন * quant_weights) {
    ভিন্ন max = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    যদি (max < GROUP_MAX_EPS) { // all zero
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) { L[i] = 0; }
        ঘূৰাই 0.f;
    }
    ভিন্ন iscale = nmax / max;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    ভিন্ন scale = 1/iscale;
    ভিন্ন best_mse = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        ভিন্ন diff = x[i] - scale*L[i];
        ভিন্ন w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    বাবে (পূর্ণাঙ্ক is = -4; is <= 4; ++is) {
        যদি (is == 0) অনুৱর্ত;
        ভিন্ন iscale_is = (0.1f*is + nmax)/max;
        ভিন্ন scale_is = 1/iscale_is;
        ভিন্ন mse = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            পূর্ণাঙ্ক l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            ভিন্ন diff = x[i] - scale_is*l;
            ভিন্ন w = quant_weights[i];
            mse += w*diff*diff;
        }
        যদি (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    ভিন্ন sumlx = 0;
    ভিন্ন suml2 = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
        পূর্ণাঙ্ক l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        ভিন্ন w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    বাবে (পূর্ণাঙ্ক itry = 0; itry < 5; ++itry) {
        পূর্ণাঙ্ক n_changed = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < n; ++i) {
            ভিন্ন w = quant_weights[i];
            ভিন্ন slx = sumlx - w*x[i]*L[i];
            ভিন্ন sl2 = suml2 - w*L[i]*L[i];
            যদি (slx > 0 && sl2 > 0) {
                পূর্ণাঙ্ক new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                যদি (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    যদি (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        যদি (!n_changed) {
            ভঙ্গ;
        }
    }
    ঘূৰাই suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

স্থির শূন্য quantize_row_q2_K_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, পূর্ণাঙ্ক k, ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights) {
    GGML_ASSERT(quant_weights);
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK_K;
    ধ্রুৱ bool requantize = true;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    ভিন্ন mins[QK_K/16];
    ভিন্ন scales[QK_K/16];
    ভিন্ন sw[QK_K/16];
    ভিন্ন weight[16];
    uint8_t Ls[QK_K/16], Lm[QK_K/16];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        memset(sw, 0, QK_K/16*পরিমাণ(ভিন্ন));
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; ++j) sumx2 += x[j]*x[j];
        ভিন্ন sigma2 = sumx2/QK_K;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            ধ্রুৱ ভিন্ন * GGML_RESTRICT qw = quant_weights + QK_K * i + 16*j;
            বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) weight[l] = qw[l] * sqrtf(sigma2 + x[16*j + l]*x[16*j + l]);
            বাবে (পূর্ণাঙ্ক l = 0; l < QK_K/16; ++l) sw[j] += weight[l];
            scales[j] = make_qkx3_quants(16, 3, x + 16*j, weight, L + 16*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        ভিন্ন dm, mm;
        dm  = make_qp_quants(QK_K/16, 15, scales, Ls, sw);
        mm  = make_qp_quants(QK_K/16, 15, mins,   Lm, sw);

        y[i].d    = GGML_FP32_TO_FP16(dm);
        y[i].dmin = GGML_FP32_TO_FP16(mm);
        dm        = GGML_FP16_TO_FP32(y[i].d);
        mm        = GGML_FP16_TO_FP32(y[i].dmin);

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            y[i].scales[j] = Ls[j] | (Lm[j] << 4);
        }

        যদি (requantize) {
            বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
                ধ্রুৱ ভিন্ন d = dm * (y[i].scales[j] & 0xF);
                যদি (!d) অনুৱর্ত;
                ধ্রুৱ ভিন্ন m = mm * (y[i].scales[j] >> 4);
                বাবে (পূর্ণাঙ্ক ii = 0; ii < 16; ++ii) {
                    পূর্ণাঙ্ক l = nearest_int((x[16*j + ii] + m)/d);
                    l = MAX(0, MIN(3, l));
                    L[16*j + ii] = l;
                }
            }
        }

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 128) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

size_t quantize_q2_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q2_K, n_per_row);
    যদি (!quant_weights) {
        quantize_row_q2_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    অন্যথা {
        ৱর্ণ * qrow = (ৱর্ণ *)dst;
        বাবে (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q2_K_impl(src, (block_q2_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    ঘূৰাই nrow * row_size;
}

//========================= 3-bit (de)-quantization

শূন্য quantize_row_q3_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK_K;

    int8_t L[QK_K];
    ভিন্ন scales[QK_K / 16];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ভিন্ন max_scale = 0;
        ভিন্ন amax = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            ভিন্ন scale = fabsf(scales[j]);
            যদি (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        যদি (max_scale) {
            ভিন্ন iscale = -32.f/max_scale;
            বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32;
                যদি (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } অন্যথা {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } অন্যথা {
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }

        int8_t sc;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * sc;
            যদি (!d) {
                অনুৱর্ত;
            }
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 16; ++ii) {
                পূর্ণাঙ্ক l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        পূর্ণাঙ্ক m = 0;
        uint8_t hm = 1;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; ++j) {
            যদি (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            যদি (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 128) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

শূন্য dequantize_row_q3_K(ধ্রুৱ block_q3_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK_K;

    ধ্রুৱ uint32_t kmask1 = 0x03030303;
    ধ্রুৱ uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    ধ্রুৱ int8_t * scales = (ধ্রুৱ int8_t*)aux;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d_all = GGML_FP16_TO_FP32(x[i].d);

        ধ্রুৱ uint8_t * GGML_RESTRICT q = x[i].qs;
        ধ্রুৱ uint8_t * GGML_RESTRICT hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        পূর্ণাঙ্ক is = 0;
        ভিন্ন dl;
        বাবে (পূর্ণাঙ্ক n = 0; n < QK_K; n += 128) {
            পূর্ণাঙ্ক shift = 0;
            বাবে (পূর্ণাঙ্ক j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}

স্থির শূন্য quantize_row_q3_K_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights) {
    প্রতিজ্ঞা(n_per_row % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = n_per_row / QK_K;

    int8_t L[QK_K];
    ভিন্ন scales[QK_K / 16];
    ভিন্ন weight[16];
    ভিন্ন sw[QK_K / 16];
    int8_t Ls[QK_K / 16];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; ++j) sumx2 += x[j]*x[j];
        ভিন্ন sigma2 = 2*sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K * i + 16*j;
                বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) weight[l] = qw[l] * sqrtf(sigma2 + x[16*j+l]*x[16*j+l]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) weight[l] = x[16*j+l]*x[16*j+l];
            }
            ভিন্ন sumw = 0;
            বাবে (পূর্ণাঙ্ক l = 0; l < 16; ++l) sumw += weight[l];
            sw[j] = sumw;

            scales[j] = make_qx_quants(16, 4, x + 16*j, L + 16*j, 1, weight);

        }

        memset(y[i].scales, 0, 12);

        ভিন্ন d_block = make_qx_quants(QK_K/16, 32, scales, Ls, 1, sw);
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            পূর্ণাঙ্ক l = Ls[j];
            যদি (j < 8) {
                y[i].scales[j] = l & 0xF;
            } অন্যথা {
                y[i].scales[j-8] |= ((l & 0xF) << 4);
            }
            l >>= 4;
            y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);

        int8_t sc;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * sc;
            যদি (!d) {
                অনুৱর্ত;
            }
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 16; ++ii) {
                পূর্ণাঙ্ক l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        পূর্ণাঙ্ক m = 0;
        uint8_t hm = 1;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; ++j) {
            যদি (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            যদি (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 128) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

size_t quantize_q3_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q3_K, n_per_row);
    যদি (!quant_weights) {
        quantize_row_q3_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    অন্যথা {
        ৱর্ণ * qrow = (ৱর্ণ *)dst;
        বাবে (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q3_K_impl(src, (block_q3_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    ঘূৰাই nrow * row_size;
}

// ====================== 4-bit (de)-quantization

শূন্য quantize_row_q4_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    ভিন্ন   weights[32];
    ভিন্ন mins[QK_K/32];
    ভিন্ন scales[QK_K/32];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন max_scale = 0; // as we are deducting the min, scales are always positive
        ভিন্ন max_min = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            ভিন্ন sum_x2 = 0;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            ভিন্ন av_x = sqrtf(sum_x2/32);
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            ভিন্ন scale = scales[j];
            যদি (scale > max_scale) {
                max_scale = scale;
            }
            ভিন্ন min = mins[j];
            যদি (min > max_min) {
                max_min = min;
            }
        }

        ভিন্ন inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        ভিন্ন inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            যদি (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } অন্যথা {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * sc;
            যদি (!d) অনুৱর্ত;
            ধ্রুৱ ভিন্ন dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 32; ++ii) {
                পূর্ণাঙ্ক l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 64) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

শূন্য dequantize_row_q4_K(ধ্রুৱ block_q4_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ পূর্ণাঙ্ক nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ uint8_t * q = x[i].qs;

        ধ্রুৱ ভিন্ন d   = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ ভিন্ন min = GGML_FP16_TO_FP32(x[i].dmin);

        পূর্ণাঙ্ক is = 0;
        uint8_t sc, m;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d1 = d * sc; ধ্রুৱ ভিন্ন m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d2 = d * sc; ধ্রুৱ ভিন্ন m2 = min * m;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

স্থির শূন্য quantize_row_q4_K_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    প্রতিজ্ঞা(n_per_row % QK_K == 0);
    ধ্রুৱ int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    ভিন্ন   weights[32];
    ভিন্ন   sw[QK_K/32];
    ভিন্ন   mins[QK_K/32];
    ভিন্ন   scales[QK_K/32];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ভিন্ন sum_x2 = 0;
        বাবে (পূর্ণাঙ্ক l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        ভিন্ন sigma2 = 2*sum_x2/QK_K;
        ভিন্ন av_x = sqrtf(sigma2);

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*i + 32*j;
                বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            }
            ভিন্ন sumw = 0;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        ভিন্ন d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        ভিন্ন m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            যদি (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } অন্যথা {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        uint8_t sc, m;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * sc;
            যদি (!d) অনুৱর্ত;
            ধ্রুৱ ভিন্ন dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 32; ++ii) {
                পূর্ণাঙ্ক l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
        uint8_t * q = y[i].qs;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 64) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;

    }
}

size_t quantize_q4_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
    যদি (!quant_weights) {
        quantize_row_q4_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    অন্যথা {
        ৱর্ণ * qrow = (ৱর্ণ *)dst;
        বাবে (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q4_K_impl(src, (block_q4_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    ঘূৰাই nrow * row_size;
}

// ====================== 5-bit (de)-quantization

শূন্য quantize_row_q5_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    uint8_t L[QK_K];
    ভিন্ন mins[QK_K/32];
    ভিন্ন scales[QK_K/32];
    ভিন্ন weights[32];
    uint8_t Laux[32];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ভিন্ন max_scale = 0; // as we are deducting the min, scales are always positive
        ভিন্ন max_min = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            ভিন্ন sum_x2 = 0;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            ভিন্ন av_x = sqrtf(sum_x2/32);
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            ভিন্ন scale = scales[j];
            যদি (scale > max_scale) {
                max_scale = scale;
            }
            ভিন্ন min = mins[j];
            যদি (min > max_min) {
                max_min = min;
            }
        }

        ভিন্ন inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        ভিন্ন inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            যদি (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } অন্যথা {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * sc;
            যদি (!d) অনুৱর্ত;
            ধ্রুৱ ভিন্ন dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 32; ++ii) {
                পূর্ণাঙ্ক l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * GGML_RESTRICT qh = y[i].qh;
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        বাবে (পূর্ণাঙ্ক n = 0; n < QK_K; n += 64) {
            বাবে (পূর্ণাঙ্ক j = 0; j < 32; ++j) {
                পূর্ণাঙ্ক l1 = L[n + j];
                যদি (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                পূর্ণাঙ্ক l2 = L[n + j + 32];
                যদি (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

শূন্য dequantize_row_q5_K(ধ্রুৱ block_q5_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ uint8_t * ql = x[i].qs;
        ধ্রুৱ uint8_t * qh = x[i].qh;

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ ভিন্ন min = GGML_FP16_TO_FP32(x[i].dmin);

        পূর্ণাঙ্ক is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d1 = d * sc; ধ্রুৱ ভিন্ন m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d2 = d * sc; ধ্রুৱ ভিন্ন m2 = min * m;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

স্থির শূন্য quantize_row_q5_K_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    প্রতিজ্ঞা(n_per_row % QK_K == 0);
    ধ্রুৱ int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    ভিন্ন   mins[QK_K/32];
    ভিন্ন   scales[QK_K/32];
    ভিন্ন   sw[QK_K/32];
    ভিন্ন   weights[32];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ভিন্ন sum_x2 = 0;
        বাবে (পূর্ণাঙ্ক l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        ভিন্ন sigma2 = 2*sum_x2/QK_K;
        ভিন্ন av_x = sqrtf(sigma2);

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*i + 32*j;
                বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            }
            ভিন্ন sumw = 0;
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;

            scales[j] = make_qkx3_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        ভিন্ন d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        ভিন্ন m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            যদি (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } অন্যথা {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        uint8_t sc, m;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * sc;
            যদি (!d) অনুৱর্ত;
            ধ্রুৱ ভিন্ন dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 32; ++ii) {
                পূর্ণাঙ্ক l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * GGML_RESTRICT qh = y[i].qh;
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        বাবে (পূর্ণাঙ্ক n = 0; n < QK_K; n += 64) {
            বাবে (পূর্ণাঙ্ক j = 0; j < 32; ++j) {
                পূর্ণাঙ্ক l1 = L[n + j];
                যদি (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                পূর্ণাঙ্ক l2 = L[n + j + 32];
                যদি (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;

    }
}

size_t quantize_q5_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_K, n_per_row);
    যদি (!quant_weights) {
        quantize_row_q5_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    অন্যথা {
        ৱর্ণ * qrow = (ৱর্ণ *)dst;
        বাবে (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q5_K_impl(src, (block_q5_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    ঘূৰাই nrow * row_size;
}

// ====================== 6-bit (de)-quantization

শূন্য quantize_row_q6_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    int8_t L[QK_K];
    ভিন্ন   scales[QK_K/16];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ভিন্ন max_scale = 0;
        ভিন্ন max_abs_scale = 0;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {

            ধ্রুৱ ভিন্ন scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, রিক্ত);
            scales[ib] = scale;

            ধ্রুৱ ভিন্ন abs_scale = fabsf(scale);
            যদি (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        যদি (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, পরিমাণ(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            অনুৱর্ত;
        }

        ভিন্ন iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            যদি (!d) {
                অনুৱর্ত;
            }
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 16; ++ii) {
                পূর্ণাঙ্ক l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 128) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) {
                ধ্রুৱ uint8_t q1 = L[j + l +  0] & 0xF;
                ধ্রুৱ uint8_t q2 = L[j + l + 32] & 0xF;
                ধ্রুৱ uint8_t q3 = L[j + l + 64] & 0xF;
                ধ্রুৱ uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

শূন্য dequantize_row_q6_K(ধ্রুৱ block_q6_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        ধ্রুৱ uint8_t * GGML_RESTRICT ql = x[i].ql;
        ধ্রুৱ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ধ্রুৱ int8_t  * GGML_RESTRICT sc = x[i].scales;

        বাবে (পূর্ণাঙ্ক n = 0; n < QK_K; n += 128) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) {
                পূর্ণাঙ্ক is = l/16;
                ধ্রুৱ int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                ধ্রুৱ int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                ধ্রুৱ int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                ধ্রুৱ int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

স্থির শূন্য quantize_row_q6_K_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    প্রতিজ্ঞা(n_per_row % QK_K == 0);
    ধ্রুৱ int64_t nb = n_per_row / QK_K;

    int8_t L[QK_K];
    ভিন্ন   scales[QK_K/16];
    //float   weights[16];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        //float sum_x2 = 0;
        //for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
        //float sigma2 = sum_x2/QK_K;

        ভিন্ন max_scale = 0;
        ভিন্ন max_abs_scale = 0;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {

            ভিন্ন scale;
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*i + 16*ib;
                //for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
                //scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
                scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, qw);
            } অন্যথা {
                scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, রিক্ত);
            }
            scales[ib] = scale;

            ধ্রুৱ ভিন্ন abs_scale = fabsf(scale);
            যদি (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        যদি (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, পরিমাণ(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            অনুৱর্ত;
        }

        ভিন্ন iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            ভিন্ন d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            যদি (!d) {
                অনুৱর্ত;
            }
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 16; ++ii) {
                পূর্ণাঙ্ক l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j += 128) {
            বাবে (পূর্ণাঙ্ক l = 0; l < 32; ++l) {
                ধ্রুৱ uint8_t q1 = L[j + l +  0] & 0xF;
                ধ্রুৱ uint8_t q2 = L[j + l + 32] & 0xF;
                ধ্রুৱ uint8_t q3 = L[j + l + 64] & 0xF;
                ধ্রুৱ uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;

    }
}

size_t quantize_q6_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
    যদি (!quant_weights) {
        quantize_row_q6_K_ref(src, dst, (int64_t)nrow*n_per_row);
    }
    অন্যথা {
        ৱর্ণ * qrow = (ৱর্ণ *)dst;
        বাবে (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q6_K_impl(src, (block_q6_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    ঘূৰাই nrow * row_size;
}

স্থির শূন্য quantize_row_q4_0_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    static_assert(QK4_0 == 32, "QK4_0 must be 32");

    যদি (!quant_weights) {
        quantize_row_q4_0_ref(x, y, n_per_row);
        ঘূৰাই;
    }

    ভিন্ন weight[QK4_0];
    int8_t L[QK4_0];

    ভিন্ন sum_x2 = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    ভিন্ন sigma2 = sum_x2/n_per_row;

    ধ্রুৱ int64_t nb = n_per_row/QK4_0;
    বাবে (পূর্ণাঙ্ক ib = 0; ib < nb; ++ib) {
        ধ্রুৱ ভিন্ন * xb = x + QK4_0 * ib;
        ধ্রুৱ ভিন্ন * qw = quant_weights + QK4_0 * ib;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK4_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        ভিন্ন d = make_qx_quants(QK4_0, 8, xb, L, 1, weight);
        y[ib].d = GGML_FP32_TO_FP16(d);
        বাবে (পূর্ণাঙ্ক j = 0; j < 16; ++j) {
            y[ib].qs[j] = L[j] | (L[j+16] << 4);
        }
    }
}

size_t quantize_q1_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    যদি (!quant_weights) {
        quantize_row_q1_0_ref(src, dst, (int64_t)nrow*n_per_row);
        ঘূৰাই nrow * ggml_row_size(GGML_TYPE_Q1_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q1_0, n_per_row);
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q1_0_ref(src, (block_q1_0*)qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    ঘূৰাই nrow * row_size;
}

size_t quantize_q2_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    যদি (!quant_weights) {
        quantize_row_q2_0_ref(src, dst, (int64_t)nrow*n_per_row);
        ঘূৰাই nrow * ggml_row_size(GGML_TYPE_Q2_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q2_0, n_per_row);
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q2_0_ref(src, (block_q2_0*)qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    ঘূৰাই nrow * row_size;
}

size_t quantize_q4_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    যদি (!quant_weights) {
        quantize_row_q4_0_ref(src, dst, (int64_t)nrow*n_per_row);
        ঘূৰাই nrow * ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q4_0_impl(src, (block_q4_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    ঘূৰাই nrow * row_size;
}

স্থির শূন্য quantize_row_q4_1_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    static_assert(QK4_1 == 32, "QK4_1 must be 32");

    যদি (!quant_weights) {
        quantize_row_q4_1_ref(x, y, n_per_row);
        ঘূৰাই;
    }

    ভিন্ন weight[QK4_1];
    uint8_t L[QK4_1], Laux[QK4_1];

    ভিন্ন sum_x2 = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    ভিন্ন sigma2 = sum_x2/n_per_row;

    ধ্রুৱ int64_t nb = n_per_row/QK4_1;
    বাবে (পূর্ণাঙ্ক ib = 0; ib < nb; ++ib) {
        ধ্রুৱ ভিন্ন * xb = x + QK4_1 * ib;
        ধ্রুৱ ভিন্ন * qw = quant_weights + QK4_1 * ib;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK4_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        ভিন্ন min;
        ভিন্ন d = make_qkx3_quants(QK4_1, 15, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
        y[ib].d = GGML_FP32_TO_FP16(d);
        y[ib].m = GGML_FP32_TO_FP16(-min);
        বাবে (পূর্ণাঙ্ক j = 0; j < 16; ++j) {
            y[ib].qs[j] = L[j] | (L[j+16] << 4);
        }
    }
}

size_t quantize_q4_1(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    যদি (!quant_weights) {
        quantize_row_q4_1_ref(src, dst, (int64_t)nrow*n_per_row);
        ঘূৰাই nrow * ggml_row_size(GGML_TYPE_Q4_1, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_1, n_per_row);
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q4_1_impl(src, (block_q4_1*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    ঘূৰাই nrow * row_size;
}

স্থির শূন্য quantize_row_q5_0_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    static_assert(QK5_0 == 32, "QK5_0 must be 32");

    যদি (!quant_weights) {
        quantize_row_q5_0_ref(x, y, n_per_row);
        ঘূৰাই;
    }

    ভিন্ন weight[QK5_0];
    int8_t L[QK5_0];

    ভিন্ন sum_x2 = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    ভিন্ন sigma2 = sum_x2/n_per_row;

    ধ্রুৱ int64_t nb = n_per_row/QK5_0;
    বাবে (পূর্ণাঙ্ক ib = 0; ib < nb; ++ib) {
        ধ্রুৱ ভিন্ন * xb = x + QK5_0 * ib;
        ধ্রুৱ ভিন্ন * qw = quant_weights + QK5_0 * ib;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK5_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        ভিন্ন d = make_qx_quants(QK5_0, 16, xb, L, 1, weight);
        y[ib].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        বাবে (পূর্ণাঙ্ক j = 0; j < 16; ++j) {
            ধ্রুৱ uint8_t xi0 = L[j];
            ধ্রুৱ uint8_t xi1 = L[j+16];
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }

        memcpy(&y[ib].qh, &qh, পরিমাণ(qh));
    }
}

size_t quantize_q5_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    যদি (!quant_weights) {
        quantize_row_q5_0_ref(src, dst, (int64_t)nrow*n_per_row);
        ঘূৰাই nrow * ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q5_0_impl(src, (block_q5_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    ঘূৰাই nrow * row_size;
}

স্থির শূন্য quantize_row_q5_1_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    static_assert(QK5_1 == 32, "QK5_1 must be 32");

    যদি (!quant_weights) {
        quantize_row_q5_1_ref(x, y, n_per_row);
        ঘূৰাই;
    }

    ভিন্ন weight[QK5_1];
    uint8_t L[QK5_1], Laux[QK5_1];

    ভিন্ন sum_x2 = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    ভিন্ন sigma2 = sum_x2/n_per_row;

    ধ্রুৱ int64_t nb = n_per_row/QK5_1;
    বাবে (পূর্ণাঙ্ক ib = 0; ib < nb; ++ib) {
        ধ্রুৱ ভিন্ন * xb = x + QK5_1 * ib;
        ধ্রুৱ ভিন্ন * qw = quant_weights + QK5_1 * ib;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK5_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        ভিন্ন min;
        ভিন্ন d = make_qkx3_quants(QK5_1, 31, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
        y[ib].d = GGML_FP32_TO_FP16(d);
        y[ib].m = GGML_FP32_TO_FP16(-min);

        uint32_t qh = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < 16; ++j) {
            ধ্রুৱ uint8_t xi0 = L[j];
            ধ্রুৱ uint8_t xi1 = L[j+16];
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }
        memcpy(&y[ib].qh, &qh, পরিমাণ(qh));
    }
}

size_t quantize_q5_1(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    যদি (!quant_weights) {
        quantize_row_q5_1_ref(src, dst, (int64_t)nrow*n_per_row);
        ঘূৰাই nrow * ggml_row_size(GGML_TYPE_Q5_1, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_1, n_per_row);
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q5_1_impl(src, (block_q5_1*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    ঘূৰাই nrow * row_size;
}

size_t quantize_q8_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    (শূন্য)quant_weights; // not used
    ধ্রুৱ size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    quantize_row_q8_0_ref(src, dst, (int64_t)nrow*n_per_row);
    ঘূৰাই nrow * row_size;
}

size_t quantize_mxfp4(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_UNUSED(quant_weights);
    quantize_row_mxfp4_ref(src, dst, (int64_t)nrow*n_per_row);
    ঘূৰাই nrow * ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
}

size_t quantize_nvfp4(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_UNUSED(quant_weights);
    quantize_row_nvfp4_ref(src, dst, (int64_t)nrow*n_per_row);
    ঘূৰাই nrow * ggml_row_size(GGML_TYPE_NVFP4, n_per_row);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

শূন্য quantize_row_tq1_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (int64_t i = 0; i < nb; i++) {
        ভিন্ন amax = 0.0f; // absolute max

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j++) {
            ধ্রুৱ ভিন্ন v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        ধ্রুৱ ভিন্ন d = amax;
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // 5 elements per byte, along 32 bytes
        বাবে (size_t j = 0; j < পরিমাণ(y->qs) - পরিমাণ(y->qs) % 32; j += 32) {
            বাবে (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                বাবে (size_t n = 0; n < 5; ++n) {
                    পূর্ণাঙ্ক xi = lroundf(x[m + n*32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*32;
        }
        // along 16 bytes
        বাবে (size_t j = পরিমাণ(y->qs) - পরিমাণ(y->qs) % 32; j < পরিমাণ(y->qs); j += 16) {
            বাবে (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                বাবে (size_t n = 0; n < 5; ++n) {
                    পূর্ণাঙ্ক xi = lroundf(x[m + n*16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*16;
        }
        // 4 elements per byte
        বাবে (size_t j = 0; j < পরিমাণ(y->qh); ++j) {
            uint8_t q = 0;
            বাবে (size_t m = 0; m < 4; ++m) {
                // -1, 0, 1 -> 0, 1, 2
                পূর্ণাঙ্ক xi = lroundf(x[j + m*পরিমাণ(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[j] = q;
        }
        x += 4*পরিমাণ(y->qh);
    }
}

শূন্য quantize_row_tq2_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (int64_t i = 0; i < nb; i++) {
        ভিন্ন amax = 0.0f; // absolute max

        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; j++) {
            ধ্রুৱ ভিন্ন v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        ধ্রুৱ ভিন্ন d = amax;
        ধ্রুৱ ভিন্ন id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        বাবে (size_t j = 0; j < পরিমাণ(y->qs); j += 32) {
            বাবে (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                বাবে (size_t n = 0; n < 4; ++n) {
                    // -1, 0, 1 -> 0, 1, 2
                    পূর্ণাঙ্ক xi = lroundf(x[m + n*32] * id) + 1;
                    q += (xi & 3) << (2*n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4*32;
        }
    }
}

size_t quantize_tq1_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    (শূন্য)quant_weights; // not used
    ধ্রুৱ size_t row_size = ggml_row_size(GGML_TYPE_TQ1_0, n_per_row);
    quantize_row_tq1_0_ref(src, dst, (int64_t)nrow*n_per_row);
    ঘূৰাই nrow * row_size;
}

size_t quantize_tq2_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    (শূন্য)quant_weights; // not used
    ধ্রুৱ size_t row_size = ggml_row_size(GGML_TYPE_TQ2_0, n_per_row);
    quantize_row_tq2_0_ref(src, dst, (int64_t)nrow*n_per_row);
    ঘূৰাই nrow * row_size;
}

শূন্য dequantize_row_tq1_0(ধ্রুৱ block_tq1_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    ধ্রুৱ uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    বাবে (int64_t i = 0; i < nb; ++i) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (size_t j = 0; j < পরিমাণ(x->qs) - পরিমাণ(x->qs) % 32; j += 32) {
            বাবে (size_t n = 0; n < 5; ++n) {
                বাবে (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (ভিন্ন) (xi - 1) * d;
                }
            }
        }
        বাবে (size_t j = পরিমাণ(x->qs) - পরিমাণ(x->qs) % 32; j < পরিমাণ(x->qs); j += 16) {
            বাবে (size_t n = 0; n < 5; ++n) {
                বাবে (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (ভিন্ন) (xi - 1) * d;
                }
            }
        }

        বাবে (size_t n = 0; n < 4; ++n) {
            বাবে (size_t j = 0; j < পরিমাণ(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (ভিন্ন) (xi - 1) * d;
            }
        }
    }
}

শূন্য dequantize_row_tq2_0(ধ্রুৱ block_tq2_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (int64_t i = 0; i < nb; ++i) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (size_t j = 0; j < পরিমাণ(x->qs); j += 32) {
            বাবে (size_t l = 0; l < 4; ++l) {
                বাবে (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (ভিন্ন) (q - 1) * d;
                }
            }
        }
    }
}

// ====================== "True" 2-bit (de)-quantization

শূন্য dequantize_row_iq2_xxs(ধ্রুৱ block_iq2_xxs * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    uint32_t aux32[2];
    ধ্রুৱ uint8_t * aux8 = (ধ্রুৱ uint8_t *)aux32;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (পূর্ণাঙ্ক ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*পরিমাণ(uint32_t));
            ধ্রুৱ ভিন্ন db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            বাবে (পূর্ণাঙ্ক l = 0; l < 4; ++l) {
                ধ্রুৱ uint8_t * grid = (ধ্রুৱ uint8_t *)(iq2xxs_grid + aux8[l]);
                ধ্রুৱ uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.3125 bpw (de)-quantization

শূন্য dequantize_row_iq2_xs(ধ্রুৱ block_iq2_xs * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    ভিন্ন db[2];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (পূর্ণাঙ্ক ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            বাবে (পূর্ণাঙ্ক l = 0; l < 4; ++l) {
                ধ্রুৱ uint8_t * grid = (ধ্রুৱ uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                ধ্রুৱ uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.5625 bpw (de)-quantization

শূন্য dequantize_row_iq2_s(ধ্রুৱ block_iq2_s * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    ভিন্ন db[2];

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ uint8_t * qs = x[i].qs;
        ধ্রুৱ uint8_t * qh = x[i].qh;
        ধ্রুৱ uint8_t * signs = qs + QK_K/8;

        বাবে (পূর্ণাঙ্ক ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            বাবে (পূর্ণাঙ্ক l = 0; l < 4; ++l) {
                ধ্রুৱ ভিন্ন dl = db[l/2];
                ধ্রুৱ uint8_t * grid = (ধ্রুৱ uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

// ====================== 3.0625 bpw (de)-quantization

শূন্য dequantize_row_iq3_xxs(ধ্রুৱ block_iq3_xxs * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    uint32_t aux32;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ uint8_t * qs = x[i].qs;
        ধ্রুৱ uint8_t * scales_and_signs = qs + QK_K/4;

        বাবে (পূর্ণাঙ্ক ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, পরিমাণ(uint32_t));
            ধ্রুৱ ভিন্ন db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            বাবে (পূর্ণাঙ্ক l = 0; l < 4; ++l) {
                ধ্রুৱ uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                ধ্রুৱ uint8_t * grid1 = (ধ্রুৱ uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                ধ্রুৱ uint8_t * grid2 = (ধ্রুৱ uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                বাবে (পূর্ণাঙ্ক j = 0; j < 4; ++j) {
                    y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

// ====================== 3.3125 bpw (de)-quantization

শূন্য dequantize_row_iq3_s(ধ্রুৱ block_iq3_s * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ uint8_t * qs = x[i].qs;
        ধ্রুৱ uint8_t * qh = x[i].qh;
        ধ্রুৱ uint8_t * signs = x[i].signs;

        বাবে (পূর্ণাঙ্ক ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            ধ্রুৱ ভিন্ন db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            ধ্রুৱ ভিন্ন db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            বাবে (পূর্ণাঙ্ক l = 0; l < 4; ++l) {
                ধ্রুৱ uint8_t * grid1 = (ধ্রুৱ uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                ধ্রুৱ uint8_t * grid2 = (ধ্রুৱ uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                বাবে (পূর্ণাঙ্ক j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            বাবে (পূর্ণাঙ্ক l = 0; l < 4; ++l) {
                ধ্রুৱ uint8_t * grid1 = (ধ্রুৱ uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                ধ্রুৱ uint8_t * grid2 = (ধ্রুৱ uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                বাবে (পূর্ণাঙ্ক j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

// ====================== 1.5625 bpw (de)-quantization

শূন্য dequantize_row_iq1_s(ধ্রুৱ block_iq1_s * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        ধ্রুৱ uint8_t  * qs = x[i].qs;
        ধ্রুৱ uint16_t * qh = x[i].qh;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/32; ++ib) {
            ধ্রুৱ ভিন্ন dl = d * (2*((qh[ib] >> 12) & 7) + 1);
            ধ্রুৱ ভিন্ন delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            বাবে (পূর্ণাঙ্ক l = 0; l < 4; ++l) {
                ধ্রুৱ int8_t * grid = (ধ্রুৱ int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

শূন্য dequantize_row_iq1_m(ধ্রুৱ block_iq1_m * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    ভিন্ন delta[4];
    uint16_t idx[4];

    iq1m_scale_t scale;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ uint16_t * sc = (ধ্রুৱ uint16_t *)x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(scale.f16);

        ধ্রুৱ uint8_t * qs = x[i].qs;
        ধ্রুৱ uint8_t * qh = x[i].qh;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/32; ++ib) {
            ধ্রুৱ ভিন্ন dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
            ধ্রুৱ ভিন্ন dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);

            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            বাবে (পূর্ণাঙ্ক l = 0; l < 2; ++l) {
                ধ্রুৱ int8_t * grid = (ধ্রুৱ int8_t *)(iq1s_grid + idx[l]);
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            বাবে (পূর্ণাঙ্ক l = 2; l < 4; ++l) {
                ধ্রুৱ int8_t * grid = (ধ্রুৱ int8_t *)(iq1s_grid + idx[l]);
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

শূন্য dequantize_row_iq4_nl(ধ্রুৱ block_iq4_nl * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK4_NL == 0);
    ধ্রুৱ int64_t nb = k / QK4_NL;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ uint8_t * qs = x[i].qs;

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);
        বাবে (পূর্ণাঙ্ক j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j+QK4_NL/2] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

শূন্য dequantize_row_iq4_xs(ধ্রুৱ block_iq4_xs * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ধ্রুৱ uint8_t * qs = x[i].qs;

        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(x[i].d);

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/32; ++ib) {
            ধ্রুৱ পূর্ণাঙ্ক ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            ধ্রুৱ ভিন্ন dl = d * (ls - 32);
            বাবে (পূর্ণাঙ্ক j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

//===================================== Q8_K ==============================================

শূন্য quantize_row_q8_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {

        ভিন্ন max = 0;
        ভিন্ন amax = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; ++j) {
            ভিন্ন ax = fabsf(x[j]);
            যদি (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        যদি (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            অনুৱর্ত;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        ধ্রুৱ ভিন্ন iscale = -127.f/max;
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; ++j) {
            পূর্ণাঙ্ক v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K/16; ++j) {
            পূর্ণাঙ্ক sum = 0;
            বাবে (পূর্ণাঙ্ক ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

শূন্য dequantize_row_q8_K(ধ্রুৱ block_q8_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    ধ্রুৱ int64_t nb = k / QK_K;

    বাবে (পূর্ণাঙ্ক i = 0; i < nb; i++) {
        বাবে (পূর্ণাঙ্ক j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

// ================================ IQ2 quantization =============================================

প্রকারনির্দেশ সংরচনা {
    uint64_t * grid;
    পূর্ণাঙ্ক      * map;
    uint16_t * neighbours;
} iq2_entry_t;

স্থির iq2_entry_t iq2_data[4] = {
    {রিক্ত, রিক্ত, রিক্ত},
    {রিক্ত, রিক্ত, রিক্ত},
    {রিক্ত, রিক্ত, রিক্ত},
    {রিক্ত, রিক্ত, রিক্ত},
};

স্থির অন্তঃস্থ পূর্ণাঙ্ক iq2_data_index(গণনা ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_S);
    ঘূৰাই type == GGML_TYPE_IQ2_XXS ? 0 :
           type == GGML_TYPE_IQ2_XS  ? 1 :
           type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? 2 : 3;
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক iq2_grid_size(গণনা ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_S);
    ঘূৰাই type == GGML_TYPE_IQ2_XXS ? 256 :
           type == GGML_TYPE_IQ2_XS  ? 512 :
           type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? NGRID_IQ1S : 1024;
}

স্থির পূর্ণাঙ্ক iq2_compare_func(ধ্রুৱ শূন্য * বাম, ধ্রুৱ শূন্য * ডান) {
    ধ্রুৱ পূর্ণাঙ্ক * l = (ধ্রুৱ পূর্ণাঙ্ক *)বাম;
    ধ্রুৱ পূর্ণাঙ্ক * r = (ধ্রুৱ পূর্ণাঙ্ক *)ডান;
    ঘূৰাই l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

শূন্য iq2xs_init_impl(গণনা ggml_type type) {
    ধ্রুৱ পূর্ণাঙ্ক gindex = iq2_data_index(type);
    ধ্রুৱ পূর্ণাঙ্ক grid_size = iq2_grid_size(type);
    যদি (iq2_data[gindex].grid) {
        ঘূৰাই;
    }
    স্থির ধ্রুৱ uint16_t kgrid_2bit_256[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };
    স্থির ধ্রুৱ uint16_t kgrid_2bit_512[512] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
          260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
          352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
          640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
         1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
         1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
         2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
         2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
         4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
         4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
         5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
         5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
         8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
         8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
        10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
        16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
        16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
        16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
        17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
        18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
        20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
        21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
        22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
        24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
        32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
        33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
        33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
        35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
        37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
        40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
        42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
    };
    স্থির ধ্রুৱ uint16_t kgrid_1bit_2048[NGRID_IQ1S] = {
            0,     2,     5,     8,    10,    17,    21,    32,    34,    40,    42,    69,    81,    84,    86,   101,
          128,   130,   136,   138,   149,   160,   162,   168,   170,   260,   261,   273,   276,   278,   281,   282,
          293,   321,   326,   329,   338,   341,   346,   353,   356,   358,   360,   389,   401,   404,   406,   421,
          512,   514,   520,   522,   533,   544,   546,   552,   554,   581,   593,   601,   612,   617,   640,   642,
          648,   650,   657,   661,   665,   672,   674,   680,   682,  1041,  1044,  1046,  1061,  1089,  1097,  1109,
         1114,  1124,  1125,  1169,  1177,  1189,  1281,  1284,  1285,  1286,  1301,  1304,  1306,  1321,  1344,  1349,
         1354,  1360,  1361,  1364,  1365,  1366,  1369,  1376,  1378,  1381,  1384,  1386,  1409,  1425,  1429,  1432,
         1434,  1441,  1444,  1445,  1446,  1449,  1556,  1561,  1601,  1604,  1616,  1618,  1621,  1624,  1632,  1633,
         1638,  1641,  1669,  1681,  1684,  1689,  2048,  2050,  2056,  2058,  2069,  2080,  2082,  2088,  2090,  2117,
         2129,  2134,  2149,  2176,  2178,  2184,  2186,  2197,  2208,  2210,  2216,  2218,  2309,  2321,  2324,  2329,
         2340,  2341,  2369,  2384,  2385,  2389,  2401,  2404,  2409,  2449,  2452,  2454,  2457,  2469,  2560,  2562,
         2568,  2570,  2581,  2592,  2594,  2600,  2602,  2629,  2641,  2649,  2657,  2661,  2688,  2690,  2693,  2696,
         2698,  2709,  2720,  2722,  2728,  2730,  4112,  4113,  4116,  4121,  4132,  4133,  4161,  4164,  4176,  4181,
         4184,  4193,  4196,  4197,  4201,  4241,  4244,  4246,  4257,  4261,  4353,  4356,  4358,  4361,  4368,  4370,
         4373,  4376,  4385,  4388,  4393,  4421,  4426,  4432,  4433,  4434,  4436,  4437,  4438,  4441,  4448,  4453,
         4484,  4498,  4501,  4513,  4516,  4625,  4628,  4630,  4645,  4672,  4678,  4681,  4690,  4693,  4696,  4698,
         4708,  4710,  4741,  4753,  4756,  4758,  4773,  5121,  5126,  5129,  5140,  5141,  5144,  5145,  5153,  5158,
         5185,  5189,  5190,  5192,  5194,  5201,  5204,  5205,  5206,  5209,  5218,  5221,  5224,  5252,  5257,  5264,
         5268,  5269,  5272,  5273,  5274,  5281,  5284,  5285,  5289,  5378,  5381,  5386,  5393,  5396,  5397,  5398,
         5401,  5408,  5410,  5413,  5416,  5418,  5441,  5444,  5445,  5446,  5457,  5458,  5460,  5461,  5462,  5465,
         5466,  5473,  5476,  5477,  5478,  5481,  5504,  5506,  5508,  5509,  5512,  5514,  5520,  5521,  5524,  5525,
         5526,  5529,  5530,  5536,  5538,  5541,  5633,  5636,  5637,  5638,  5653,  5654,  5656,  5658,  5665,  5670,
         5696,  5698,  5700,  5701,  5704,  5706,  5713,  5717,  5718,  5720,  5721,  5729,  5732,  5733,  5736,  5737,
         5738,  5766,  5770,  5778,  5781,  5796,  5801,  6161,  6166,  6181,  6209,  6212,  6214,  6217,  6224,  6229,
         6232,  6234,  6240,  6241,  6244,  6246,  6249,  6277,  6289,  6292,  6309,  6416,  6418,  6421,  6426,  6433,
         6437,  6466,  6468,  6469,  6472,  6481,  6484,  6485,  6486,  6489,  6490,  6496,  6501,  6506,  6537,  6545,
         6546,  6549,  6552,  6561,  6566,  6569,  6665,  6678,  6692,  6694,  6724,  6726,  6729,  6736,  6738,  6741,
         6744,  6753,  6758,  6761,  6789,  6801,  6806,  6810,  8192,  8194,  8200,  8202,  8213,  8224,  8226,  8229,
         8232,  8234,  8261,  8273,  8281,  8289,  8293,  8320,  8322,  8328,  8330,  8341,  8352,  8354,  8357,  8360,
         8362,  8453,  8465,  8468,  8473,  8485,  8514,  8516,  8521,  8533,  8536,  8538,  8545,  8548,  8549,  8550,
         8581,  8592,  8598,  8601,  8613,  8705,  8712,  8714,  8721,  8725,  8736,  8738,  8744,  8746,  8773,  8785,
         8790,  8793,  8805,  8833,  8840,  8842,  8849,  8853,  8864,  8866,  8872,  8874,  9221,  9236,  9238,  9241,
         9253,  9284,  9285,  9286,  9289,  9298,  9301,  9304,  9306,  9318,  9349,  9361,  9364,  9369,  9377,  9381,
         9481,  9493,  9505,  9513,  9536,  9541,  9544,  9553,  9556,  9557,  9561,  9570,  9573,  9576,  9609,  9616,
         9620,  9621,  9624,  9626,  9633,  9636,  9638,  9641,  9733,  9744,  9746,  9753,  9765,  9793,  9801,  9813,
         9824,  9825,  9833,  9860,  9862,  9872,  9882, 10240, 10242, 10248, 10250, 10261, 10272, 10274, 10280, 10282,
        10309, 10321, 10324, 10341, 10368, 10370, 10376, 10378, 10400, 10402, 10408, 10410, 10505, 10513, 10516, 10521,
        10533, 10566, 10569, 10578, 10581, 10593, 10596, 10598, 10601, 10629, 10640, 10646, 10649, 10660, 10661, 10752,
        10754, 10760, 10762, 10784, 10786, 10792, 10794, 10821, 10833, 10838, 10841, 10853, 10880, 10882, 10888, 10890,
        10901, 10912, 10914, 10920, 10922, 16389, 16401, 16406, 16421, 16457, 16466, 16469, 16472, 16474, 16481, 16484,
        16486, 16532, 16537, 16545, 16550, 16640, 16641, 16644, 16646, 16649, 16658, 16661, 16662, 16664, 16666, 16673,
        16678, 16681, 16709, 16712, 16714, 16721, 16724, 16725, 16726, 16729, 16730, 16741, 16744, 16746, 16769, 16772,
        16774, 16784, 16786, 16789, 16800, 16801, 16802, 16901, 16913, 16916, 16918, 16933, 16961, 16978, 16981, 16986,
        16996, 17001, 17033, 17044, 17061, 17409, 17429, 17433, 17449, 17477, 17480, 17482, 17489, 17492, 17493, 17494,
        17505, 17506, 17509, 17512, 17514, 17537, 17542, 17545, 17552, 17554, 17557, 17568, 17569, 17577, 17665, 17666,
        17669, 17674, 17681, 17684, 17685, 17686, 17689, 17696, 17701, 17706, 17729, 17732, 17733, 17734, 17737, 17744,
        17745, 17748, 17749, 17750, 17752, 17753, 17761, 17764, 17765, 17766, 17769, 17794, 17796, 17797, 17800, 17809,
        17812, 17813, 17814, 17817, 17818, 17829, 17832, 17834, 17921, 17925, 17929, 17940, 17941, 17944, 17946, 17953,
        17956, 17961, 17984, 17986, 17989, 17992, 18000, 18001, 18002, 18005, 18006, 18009, 18018, 18021, 18024, 18049,
        18053, 18058, 18068, 18069, 18081, 18084, 18086, 18437, 18449, 18453, 18458, 18469, 18498, 18505, 18512, 18517,
        18520, 18529, 18532, 18534, 18537, 18565, 18577, 18580, 18582, 18585, 18597, 18689, 18693, 18694, 18698, 18704,
        18708, 18709, 18712, 18721, 18724, 18726, 18752, 18757, 18762, 18769, 18770, 18772, 18773, 18774, 18777, 18784,
        18786, 18789, 18790, 18794, 18822, 18825, 18834, 18837, 18838, 18840, 18849, 18852, 18854, 18857, 18966, 19012,
        19014, 19017, 19029, 19032, 19034, 19044, 19049, 19092, 19109, 20481, 20484, 20485, 20486, 20489, 20498, 20501,
        20506, 20513, 20516, 20521, 20544, 20549, 20552, 20561, 20564, 20565, 20566, 20569, 20581, 20584, 20614, 20617,
        20629, 20632, 20640, 20641, 20646, 20649, 20741, 20744, 20745, 20746, 20753, 20756, 20757, 20758, 20760, 20761,
        20768, 20773, 20774, 20776, 20778, 20801, 20804, 20805, 20806, 20809, 20816, 20817, 20818, 20820, 20821, 20822,
        20824, 20825, 20826, 20833, 20836, 20837, 20838, 20841, 20866, 20869, 20881, 20884, 20885, 20886, 20889, 20896,
        20901, 20906, 20993, 20998, 21010, 21013, 21018, 21025, 21028, 21058, 21061, 21066, 21073, 21076, 21077, 21078,
        21081, 21090, 21093, 21125, 21136, 21138, 21141, 21145, 21146, 21156, 21508, 21509, 21521, 21524, 21525, 21526,
        21528, 21529, 21537, 21541, 21544, 21546, 21569, 21572, 21573, 21574, 21577, 21578, 21584, 21585, 21588, 21589,
        21590, 21592, 21593, 21594, 21601, 21602, 21604, 21605, 21606, 21609, 21632, 21640, 21642, 21649, 21652, 21653,
        21654, 21657, 21665, 21668, 21669, 21674, 21761, 21762, 21764, 21765, 21766, 21769, 21776, 21777, 21778, 21780,
        21781, 21782, 21785, 21786, 21793, 21796, 21797, 21798, 21801, 21824, 21825, 21826, 21828, 21829, 21830, 21832,
        21833, 21840, 21841, 21842, 21844, 21845, 21846, 21848, 21849, 21850, 21856, 21857, 21860, 21861, 21862, 21864,
        21865, 21866, 21889, 21892, 21893, 21897, 21898, 21904, 21905, 21908, 21909, 21910, 21912, 21913, 21921, 21924,
        21925, 21926, 21929, 22016, 22017, 22018, 22020, 22022, 22024, 22025, 22033, 22036, 22037, 22040, 22041, 22048,
        22049, 22050, 22052, 22053, 22054, 22056, 22057, 22081, 22085, 22086, 22088, 22089, 22090, 22096, 22097, 22098,
        22100, 22101, 22102, 22104, 22105, 22106, 22113, 22116, 22117, 22121, 22146, 22149, 22150, 22152, 22153, 22154,
        22161, 22165, 22170, 22178, 22181, 22182, 22184, 22185, 22532, 22533, 22534, 22537, 22544, 22549, 22552, 22561,
        22570, 22597, 22600, 22602, 22609, 22612, 22613, 22614, 22616, 22617, 22624, 22626, 22628, 22629, 22658, 22665,
        22672, 22674, 22677, 22680, 22689, 22697, 22785, 22786, 22789, 22794, 22801, 22804, 22805, 22806, 22809, 22821,
        22849, 22852, 22853, 22854, 22857, 22864, 22865, 22866, 22868, 22869, 22870, 22872, 22873, 22874, 22881, 22884,
        22885, 22886, 22889, 22913, 22917, 22921, 22929, 22932, 22933, 22934, 22936, 22937, 22949, 23044, 23048, 23061,
        23066, 23072, 23077, 23078, 23081, 23109, 23112, 23113, 23121, 23125, 23126, 23128, 23129, 23138, 23141, 23144,
        23146, 23169, 23178, 23186, 23189, 23190, 23192, 23194, 23201, 24581, 24596, 24598, 24601, 24613, 24644, 24656,
        24661, 24662, 24664, 24666, 24673, 24676, 24678, 24681, 24705, 24726, 24741, 24833, 24836, 24838, 24841, 24850,
        24853, 24865, 24866, 24870, 24873, 24901, 24905, 24913, 24917, 24918, 24921, 24933, 24934, 24938, 24964, 24970,
        24978, 24981, 24993, 24998, 25001, 25105, 25110, 25113, 25152, 25153, 25158, 25173, 25174, 25176, 25184, 25221,
        25233, 25238, 25253, 25617, 25618, 25621, 25622, 25626, 25633, 25638, 25641, 25664, 25666, 25669, 25672, 25674,
        25681, 25684, 25685, 25686, 25689, 25690, 25696, 25698, 25701, 25732, 25733, 25737, 25744, 25746, 25748, 25749,
        25750, 25752, 25754, 25761, 25764, 25769, 25861, 25864, 25866, 25873, 25877, 25878, 25881, 25924, 25925, 25926,
        25929, 25936, 25937, 25940, 25941, 25942, 25945, 25953, 25956, 25957, 25958, 25961, 25990, 25993, 25994, 26001,
        26005, 26006, 26009, 26010, 26018, 26021, 26022, 26024, 26114, 26121, 26133, 26144, 26150, 26152, 26153, 26176,
        26181, 26184, 26186, 26193, 26196, 26197, 26198, 26200, 26202, 26208, 26213, 26216, 26240, 26242, 26245, 26250,
        26260, 26262, 26264, 26265, 26272, 26276, 26278, 26282, 26646, 26649, 26661, 26689, 26706, 26709, 26714, 26721,
        26729, 26757, 26769, 26776, 26790, 26881, 26884, 26896, 26901, 26913, 26916, 26918, 26921, 26944, 26945, 26949,
        26950, 26952, 26961, 26964, 26965, 26966, 26969, 26976, 26981, 26986, 27010, 27012, 27018, 27029, 27041, 27044,
        27045, 27049, 27153, 27158, 27160, 27201, 27204, 27209, 27216, 27221, 27224, 27226, 27236, 27237, 27241, 27270,
        27284, 27288, 27290, 27302, 32768, 32770, 32776, 32778, 32800, 32802, 32808, 32810, 32837, 32848, 32849, 32852,
        32854, 32857, 32869, 32896, 32898, 32904, 32906, 32917, 32928, 32930, 32936, 32938, 33029, 33041, 33044, 33046,
        33049, 33061, 33089, 33092, 33097, 33104, 33106, 33109, 33110, 33112, 33113, 33124, 33126, 33129, 33157, 33161,
        33172, 33174, 33177, 33189, 33280, 33282, 33288, 33290, 33301, 33312, 33314, 33320, 33322, 33361, 33364, 33369,
        33381, 33408, 33410, 33416, 33418, 33429, 33440, 33442, 33448, 33450, 33812, 33817, 33857, 33860, 33873, 33877,
        33882, 33889, 33892, 33897, 33940, 33945, 34049, 34057, 34066, 34069, 34074, 34086, 34089, 34112, 34113, 34117,
        34120, 34129, 34132, 34133, 34134, 34137, 34138, 34149, 34150, 34152, 34154, 34177, 34180, 34182, 34185, 34192,
        34194, 34197, 34200, 34214, 34321, 34326, 34329, 34341, 34369, 34372, 34377, 34378, 34384, 34389, 34393, 34394,
        34401, 34406, 34410, 34437, 34449, 34458, 34468, 34816, 34818, 34824, 34826, 34837, 34848, 34850, 34856, 34858,
        34881, 34885, 34897, 34900, 34905, 34917, 34921, 34944, 34946, 34952, 34954, 34965, 34976, 34978, 34984, 34986,
        35077, 35078, 35089, 35092, 35094, 35109, 35137, 35140, 35142, 35145, 35152, 35154, 35157, 35162, 35169, 35172,
        35205, 35222, 35225, 35237, 35328, 35330, 35336, 35338, 35349, 35360, 35362, 35368, 35370, 35397, 35409, 35412,
        35414, 35456, 35458, 35464, 35466, 35477, 35488, 35490, 35496, 35498, 36869, 36881, 36886, 36888, 36889, 36901,
        36929, 36934, 36937, 36949, 36952, 36954, 36969, 36970, 36997, 37009, 37012, 37014, 37017, 37029, 37121, 37124,
        37126, 37129, 37136, 37141, 37144, 37146, 37153, 37156, 37158, 37161, 37184, 37189, 37200, 37201, 37204, 37205,
        37206, 37209, 37218, 37221, 37252, 37254, 37266, 37269, 37272, 37281, 37284, 37286, 37289, 37381, 37393, 37396,
        37401, 37413, 37444, 37446, 37449, 37456, 37458, 37461, 37464, 37478, 37481, 37509, 37524, 37526, 37545, 37889,
        37892, 37894, 37904, 37909, 37912, 37926, 37952, 37962, 37969, 37972, 37973, 37974, 37976, 37977, 37984, 37985,
        37986, 37989, 38020, 38022, 38034, 38036, 38037, 38040, 38049, 38057, 38144, 38149, 38152, 38154, 38160, 38161,
        38164, 38165, 38166, 38169, 38177, 38181, 38185, 38186, 38209, 38212, 38213, 38214, 38217, 38224, 38225, 38226,
        38228, 38229, 38230, 38232, 38233, 38234, 38241, 38244, 38245, 38246, 38249, 38273, 38277, 38280, 38289, 38290,
        38292, 38293, 38294, 38297, 38298, 38304, 38306, 38309, 38312, 38314, 38401, 38404, 38416, 38421, 38425, 38432,
        38438, 38441, 38469, 38472, 38473, 38481, 38482, 38485, 38486, 38489, 38501, 38504, 38530, 38532, 38537, 38538,
        38546, 38548, 38549, 38564, 38566, 38569, 38917, 38934, 38937, 38949, 38977, 38982, 38992, 38994, 38997, 38998,
        39002, 39012, 39013, 39045, 39057, 39062, 39065, 39077, 39172, 39174, 39177, 39184, 39186, 39189, 39192, 39194,
        39200, 39201, 39204, 39206, 39232, 39234, 39237, 39240, 39242, 39249, 39252, 39253, 39254, 39257, 39266, 39269,
        39270, 39274, 39297, 39300, 39312, 39314, 39317, 39322, 39329, 39334, 39429, 39445, 39461, 39492, 39494, 39497,
        39504, 39509, 39512, 39521, 39557, 39569, 39572, 39573, 39574, 40960, 40962, 40968, 40970, 40981, 40992, 40994,
        41000, 41002, 41029, 41041, 41044, 41046, 41049, 41088, 41090, 41096, 41098, 41109, 41120, 41122, 41128, 41130,
        41221, 41225, 41233, 41236, 41238, 41241, 41242, 41286, 41289, 41297, 41301, 41304, 41306, 41313, 41316, 41349,
        41360, 41362, 41366, 41369, 41474, 41480, 41482, 41488, 41497, 41506, 41512, 41514, 41541, 41553, 41558, 41561,
        41573, 41600, 41602, 41608, 41610, 41621, 41632, 41634, 41640, 41642, 42009, 42021, 42049, 42052, 42064, 42068,
        42069, 42072, 42074, 42081, 42085, 42086, 42088, 42089, 42117, 42246, 42249, 42256, 42258, 42261, 42264, 42278,
        42281, 42306, 42309, 42321, 42324, 42325, 42326, 42329, 42341, 42346, 42369, 42372, 42373, 42374, 42377, 42386,
        42389, 42392, 42501, 42513, 42518, 42522, 42529, 42533, 42564, 42566, 42570, 42578, 42581, 42582, 42584, 42592,
        42594, 42630, 42640, 42645, 42646, 42649, 42657, 42660, 42662, 43008, 43010, 43016, 43018, 43040, 43042, 43048,
        43050, 43089, 43092, 43094, 43097, 43136, 43138, 43144, 43146, 43157, 43168, 43170, 43176, 43178, 43269, 43284,
        43289, 43297, 43301, 43329, 43344, 43349, 43354, 43361, 43366, 43369, 43408, 43414, 43520, 43522, 43528, 43530,
        43552, 43554, 43560, 43562, 43601, 43604, 43606, 43648, 43650, 43656, 43658, 43669, 43680, 43682, 43688, 43690,
    };
    স্থির ধ্রুৱ uint16_t kgrid_2bit_1024[1024] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   102,   105,   128,   130,   133,   136,   145,   148,   160,
          165,   170,   257,   260,   262,   265,   272,   274,   277,   280,   289,   292,   320,   322,   325,   328,
          337,   340,   342,   345,   352,   357,   360,   385,   388,   400,   402,   405,   417,   420,   512,   514,
          517,   520,   529,   532,   544,   554,   577,   580,   582,   585,   592,   597,   640,   645,   650,   660,
          674,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1062,  1065,  1088,  1090,  1093,
         1096,  1098,  1105,  1108,  1110,  1113,  1120,  1122,  1125,  1153,  1156,  1158,  1161,  1168,  1173,  1176,
         1185,  1188,  1280,  1282,  1285,  1288,  1290,  1297,  1300,  1302,  1305,  1312,  1317,  1320,  1345,  1348,
         1350,  1353,  1360,  1362,  1365,  1368,  1377,  1380,  1408,  1410,  1413,  1416,  1425,  1428,  1440,  1537,
         1540,  1542,  1545,  1552,  1557,  1600,  1605,  1608,  1617,  1620,  1632,  1665,  1668,  1680,  2048,  2050,
         2053,  2056,  2065,  2068,  2070,  2073,  2080,  2085,  2090,  2113,  2116,  2118,  2121,  2128,  2130,  2133,
         2136,  2145,  2148,  2176,  2181,  2196,  2218,  2305,  2308,  2320,  2322,  2325,  2328,  2337,  2368,  2373,
         2376,  2385,  2388,  2400,  2433,  2448,  2560,  2577,  2580,  2594,  2600,  2602,  2640,  2713,  4097,  4100,
         4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4134,  4160,  4162,  4165,  4168,  4177,  4180,  4182,
         4185,  4192,  4194,  4197,  4200,  4225,  4228,  4230,  4240,  4245,  4248,  4257,  4260,  4352,  4354,  4357,
         4360,  4362,  4369,  4372,  4374,  4377,  4384,  4386,  4389,  4392,  4417,  4420,  4422,  4425,  4432,  4434,
         4437,  4440,  4449,  4452,  4480,  4482,  4485,  4488,  4497,  4500,  4609,  4612,  4617,  4624,  4629,  4641,
         4644,  4672,  4677,  4689,  4692,  4737,  4740,  4752,  5120,  5122,  5125,  5128,  5137,  5140,  5142,  5145,
         5152,  5157,  5160,  5185,  5188,  5190,  5193,  5200,  5202,  5205,  5208,  5217,  5220,  5248,  5250,  5253,
         5256,  5265,  5268,  5280,  5377,  5380,  5382,  5385,  5392,  5394,  5397,  5400,  5409,  5412,  5440,  5442,
         5445,  5448,  5457,  5460,  5472,  5505,  5508,  5520,  5632,  5637,  5640,  5649,  5652,  5664,  5697,  5700,
         5712,  5760,  5802,  6145,  6148,  6150,  6153,  6160,  6165,  6168,  6177,  6208,  6210,  6213,  6216,  6225,
         6228,  6240,  6273,  6276,  6400,  6402,  6405,  6408,  6417,  6420,  6432,  6465,  6468,  6480,  6505,  6562,
         6660,  6672,  6720,  6742,  8192,  8194,  8197,  8200,  8209,  8212,  8214,  8217,  8224,  8229,  8234,  8257,
         8260,  8272,  8274,  8277,  8292,  8320,  8330,  8340,  8362,  8449,  8452,  8464,  8466,  8469,  8481,  8512,
         8514,  8517,  8529,  8532,  8544,  8577,  8580,  8592,  8704,  8714,  8738,  8744,  8746,  8772,  8784,  8840,
         8842,  8872,  9217,  9220,  9222,  9225,  9232,  9237,  9240,  9249,  9252,  9280,  9282,  9285,  9288,  9297,
         9300,  9312,  9345,  9348,  9360,  9472,  9477,  9480,  9489,  9492,  9504,  9537,  9540,  9552,  9574,  9600,
         9729,  9732,  9744,  9792,  9817, 10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497, 10500,
        10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16410,
        16417, 16420, 16422, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470, 16473, 16480, 16482, 16485, 16513,
        16516, 16528, 16533, 16536, 16545, 16548, 16640, 16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674,
        16677, 16705, 16708, 16710, 16713, 16720, 16722, 16725, 16728, 16737, 16740, 16768, 16770, 16773, 16776, 16785,
        16788, 16800, 16897, 16900, 16912, 16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992, 17025,
        17028, 17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430, 17433, 17440, 17442, 17445, 17448, 17473, 17476,
        17478, 17481, 17488, 17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544, 17553, 17556, 17568, 17665,
        17668, 17670, 17673, 17680, 17682, 17685, 17688, 17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760,
        17770, 17793, 17796, 17808, 17920, 17922, 17925, 17928, 17937, 17940, 17952, 17985, 17988, 18000, 18048, 18085,
        18433, 18436, 18441, 18448, 18450, 18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516, 18528,
        18564, 18576, 18688, 18690, 18693, 18696, 18705, 18708, 18720, 18753, 18756, 18768, 18816, 18838, 18945, 18948,
        18960, 19008, 20480, 20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517, 20520, 20545, 20548,
        20550, 20553, 20560, 20562, 20565, 20568, 20577, 20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740,
        20742, 20745, 20752, 20754, 20757, 20760, 20769, 20772, 20800, 20802, 20805, 20808, 20817, 20820, 20832, 20865,
        20868, 20880, 20992, 20997, 21000, 21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508, 21510,
        21513, 21520, 21522, 21525, 21528, 21537, 21540, 21568, 21570, 21573, 21576, 21585, 21588, 21600, 21633, 21636,
        21648, 21760, 21762, 21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017, 22020, 22032, 22054,
        22080, 22528, 22530, 22533, 22536, 22545, 22548, 22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800,
        22848, 23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592, 24594, 24597, 24600, 24609, 24612, 24640, 24645,
        24648, 24657, 24660, 24672, 24708, 24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900, 24912,
        24960, 24985, 25092, 25104, 25152, 25174, 25249, 25600, 25605, 25608, 25617, 25620, 25632, 25665, 25668, 25680,
        25728, 25857, 25860, 25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640, 26725, 26776, 26880,
        26922, 27202, 27297, 32768, 32770, 32773, 32776, 32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850,
        32853, 32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033, 33040, 33042, 33045, 33048, 33057, 33060,
        33088, 33090, 33093, 33096, 33105, 33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300, 33345,
        33348, 33360, 33793, 33796, 33798, 33801, 33808, 33810, 33813, 33816, 33825, 33856, 33858, 33861, 33864, 33873,
        33876, 33888, 33921, 33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113, 34116, 34128, 34176,
        34186, 34305, 34308, 34320, 34345, 34368, 34816, 34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076,
        35136, 35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880, 36882, 36885, 36888, 36900, 36928,
        36930, 36933, 36936, 36945, 36948, 36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188, 37200,
        37210, 37377, 37380, 37392, 37440, 37542, 37888, 37890, 37893, 37896, 37905, 37908, 37920, 37953, 37956, 37968,
        38016, 38038, 38145, 38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916, 38928, 38950, 38976,
        39081, 39168, 39241, 39250, 39568, 40960, 40965, 40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130,
        41280, 41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640, 41985, 41988, 42000, 42048, 42121,
        42148, 42240, 42265, 42577, 43018, 43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656, 43690,
    };

    ধ্রুৱ পূর্ণাঙ্ক kmap_size = 43692;
    //const int nwant = type == GGML_TYPE_IQ1_S ? 3 : 2;
    ধ্রুৱ পূর্ণাঙ্ক nwant = type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? 3 : type == GGML_TYPE_IQ2_S ? 1 : 2;
    ধ্রুৱ uint16_t * kgrid = type == GGML_TYPE_IQ2_XXS ? kgrid_2bit_256 :
                             type == GGML_TYPE_IQ2_XS  ? kgrid_2bit_512 :
                             type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? kgrid_1bit_2048 : kgrid_2bit_1024;
    uint64_t * kgrid_q2xs;
    পূর্ণাঙ্ক      * kmap_q2xs;
    uint16_t * kneighbors_q2xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);
    uint64_t * the_grid = (uint64_t *)malloc(grid_size*পরিমাণ(uint64_t));
    বাবে (পূর্ণাঙ্ক k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
            পূর্ণাঙ্ক l = (kgrid[k] >> 2*i) & 0x3;
            pos[i] = 2*l + 1;
        }
    }
    kgrid_q2xs = the_grid;
    iq2_data[gindex].grid = the_grid;
    kmap_q2xs = (পূর্ণাঙ্ক *)malloc(kmap_size*পরিমাণ(পূর্ণাঙ্ক));
    iq2_data[gindex].map = kmap_q2xs;
    বাবে (পূর্ণাঙ্ক i = 0; i < kmap_size; ++i) kmap_q2xs[i] = -1;
    uint64_t aux64;
    uint8_t * aux8 = (uint8_t *)&aux64;
    বাবে (পূর্ণাঙ্ক i = 0; i < grid_size; ++i) {
        aux64 = kgrid_q2xs[i];
        uint16_t index = 0;
        বাবে (পূর্ণাঙ্ক k=0; k<8; ++k) {
            uint16_t q = (aux8[k] - 1)/2;
            index |= (q << 2*k);
        }
        kmap_q2xs[index] = i;
    }
    // The neighbour search runs in three passes:
    //   1. Parallel: for each i, qsort and count its neighbours into n_per_i,
    //      and reduce the totals (num_neighbors, num_not_in_map).
    //   2. Serial: prefix-sum n_per_i into offsets[], so each i has a
    //      pre-assigned slice of kneighbors_q2xs to write into.
    //   3. Parallel: redo the qsort and write each i's neighbour list at
    //      offsets[i].
    পূর্ণাঙ্ক * n_per_i = (পূর্ণাঙ্ক *)malloc(kmap_size*পরিমাণ(পূর্ণাঙ্ক));
    GGML_ASSERT(n_per_i);
    পূর্ণাঙ্ক অঙ্ক_neighbors = 0, অঙ্ক_not_in_map = 0;
#ifdef GGML_USE_OPENMP
    #pragma omp parallel reduction(+:অঙ্ক_neighbors,অঙ্ক_not_in_map)
#endif
    {
        পূর্ণাঙ্ক * dist2 = (পূর্ণাঙ্ক *)malloc(2*grid_size*পরিমাণ(পূর্ণাঙ্ক));
        GGML_ASSERT(dist2);
        int8_t pos[8];
        পূর্ণাঙ্ক i;
#ifdef GGML_USE_OPENMP
        #pragma omp বাবে schedule(dynamic, 64)
#endif
        বাবে (i = 0; i < kmap_size; ++i) {
            যদি (kmap_q2xs[i] >= 0) {
                n_per_i[i] = 0;
                অনুৱর্ত;
            }
            ++অঙ্ক_not_in_map;
            বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) {
                পূর্ণাঙ্ক l = (i >> 2*k) & 0x3;
                pos[k] = 2*l + 1;
            }
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q2xs + j);
                পূর্ণাঙ্ক d2 = 0;
                বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
                dist2[2*j+0] = d2;
                dist2[2*j+1] = j;
            }
            qsort(dist2, grid_size, 2*পরিমাণ(পূর্ণাঙ্ক), iq2_compare_func);
            পূর্ণাঙ্ক n = 0; পূর্ণাঙ্ক d2 = dist2[0];
            পূর্ণাঙ্ক nhave = 1;
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                যদি (dist2[2*j] > d2) {
                    যদি (nhave == nwant) ভঙ্গ;
                    d2 = dist2[2*j];
                    ++nhave;
                }
                ++n;
            }
            n_per_i[i] = n;
            অঙ্ক_neighbors += n;
        }
        মুক্ত(dist2);
    }
    //printf("%s: %d neighbours in total\n", __func__, num_neighbors);
    kneighbors_q2xs = (uint16_t *)malloc((অঙ্ক_neighbors + অঙ্ক_not_in_map)*পরিমাণ(uint16_t));
    iq2_data[gindex].neighbours = kneighbors_q2xs;

    পূর্ণাঙ্ক * offsets = (পূর্ণাঙ্ক *)malloc(kmap_size*পরিমাণ(পূর্ণাঙ্ক));
    GGML_ASSERT(offsets);
    পূর্ণাঙ্ক counter = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < kmap_size; ++i) {
        যদি (kmap_q2xs[i] >= 0) {
            offsets[i] = -1;
            অনুৱর্ত;
        }
        offsets[i] = counter;
        counter += 1 + n_per_i[i];
    }

#ifdef GGML_USE_OPENMP
    #pragma omp parallel
#endif
    {
        পূর্ণাঙ্ক * dist2 = (পূর্ণাঙ্ক *)malloc(2*grid_size*পরিমাণ(পূর্ণাঙ্ক));
        GGML_ASSERT(dist2);
        int8_t pos[8];
        পূর্ণাঙ্ক i;
#ifdef GGML_USE_OPENMP
        #pragma omp বাবে schedule(dynamic, 64)
#endif
        বাবে (i = 0; i < kmap_size; ++i) {
            যদি (kmap_q2xs[i] >= 0) অনুৱর্ত;
            বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) {
                পূর্ণাঙ্ক l = (i >> 2*k) & 0x3;
                pos[k] = 2*l + 1;
            }
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q2xs + j);
                পূর্ণাঙ্ক d2 = 0;
                বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
                dist2[2*j+0] = d2;
                dist2[2*j+1] = j;
            }
            qsort(dist2, grid_size, 2*পরিমাণ(পূর্ণাঙ্ক), iq2_compare_func);
            পূর্ণাঙ্ক local_counter = offsets[i];
            kmap_q2xs[i] = -(local_counter + 1);
            পূর্ণাঙ্ক d2 = dist2[0];
            uint16_t * start = &kneighbors_q2xs[local_counter++];
            পূর্ণাঙ্ক n = 0, nhave = 1;
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                যদি (dist2[2*j] > d2) {
                    যদি (nhave == nwant) ভঙ্গ;
                    d2 = dist2[2*j];
                    ++nhave;
                }
                kneighbors_q2xs[local_counter++] = dist2[2*j+1];
                ++n;
            }
            *start = n;
        }
        মুক্ত(dist2);
    }
    মুক্ত(offsets);
    মুক্ত(n_per_i);
}

শূন্য iq2xs_free_impl(গণনা ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_S);
    ধ্রুৱ পূর্ণাঙ্ক gindex = iq2_data_index(type);
    যদি (iq2_data[gindex].grid) {
        মুক্ত(iq2_data[gindex].grid);       iq2_data[gindex].grid = রিক্ত;
        মুক্ত(iq2_data[gindex].map);        iq2_data[gindex].map  = রিক্ত;
        মুক্ত(iq2_data[gindex].neighbours); iq2_data[gindex].neighbours = রিক্ত;
    }
}

স্থির পূর্ণাঙ্ক iq2_find_best_neighbour(ধ্রুৱ uint16_t * GGML_RESTRICT neighbours, ধ্রুৱ uint64_t * GGML_RESTRICT grid,
        ধ্রুৱ ভিন্ন * GGML_RESTRICT xval, ধ্রুৱ ভিন্ন * GGML_RESTRICT weight, ভিন্ন scale, int8_t * GGML_RESTRICT L) {
    পূর্ণাঙ্ক অঙ্ক_neighbors = neighbours[0];
    GGML_ASSERT(অঙ্ক_neighbors > 0);
    ভিন্ন best_d2 = FLT_MAX;
    পূর্ণাঙ্ক grid_index = -1;
    বাবে (পূর্ণাঙ্ক j = 1; j <= অঙ্ক_neighbors; ++j) {
        ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + neighbours[j]);
        ভিন্ন d2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
            ভিন্ন q = pg[i];
            ভিন্ন diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        যদি (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + grid_index);
    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    ঘূৰাই grid_index;
}

স্থির শূন্য quantize_row_iq2_xxs_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t n, ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights) {

    ধ্রুৱ পূর্ণাঙ্ক gindex = iq2_data_index(GGML_TYPE_IQ2_XXS);

    ধ্রুৱ uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    ধ্রুৱ পূর্ণাঙ্ক      * kmap_q2xs       = iq2_data[gindex].map;
    ধ্রুৱ uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    ধ্রুৱ পূর্ণাঙ্ক kMaxQ = 3;

    ধ্রুৱ int64_t nbl = n/QK_K;

    block_iq2_xxs * y = vy;

    ভিন্ন scales[QK_K/32];
    ভিন্ন weight[32];
    ভিন্ন xval[32];
    int8_t L[32];
    int8_t Laux[32];
    ভিন্ন  waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];

    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);

        ভিন্ন max_scale = 0;

        ধ্রুৱ ভিন্ন * xbl = x + QK_K*ibl;
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        ভিন্ন sigma2 = sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/32; ++ib) {
            ধ্রুৱ ভিন্ন * xb = xbl + 32*ib;
            ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + 32*ib;
            বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                পূর্ণাঙ্ক nflip = 0;
                uint8_t s = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                    যদি (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    অন্যথা {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                যদি (nflip%2) {
                    পূর্ণাঙ্ক imin = 0; ভিন্ন min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    বাবে (পূর্ণাঙ্ক i = 1; i < 8; ++i) {
                        ভিন্ন ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        যদি (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            ভিন্ন max = xval[0];
            বাবে (পূর্ণাঙ্ক i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            যদি (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                অনুৱর্ত;
            }
            ভিন্ন scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
            ভিন্ন eff_max = scale*kMaxQ;
            যদি (eff_max <= 0) {
                scales[ib] = 0;
                memset(L, 0, 32);
                অনুৱর্ত;
            }
            ভিন্ন best = 0;
            বাবে (পূর্ণাঙ্ক is = -6; is <= 6; ++is) {
                ভিন্ন id = (2*kMaxQ-1+is*0.1f)/eff_max;
                ভিন্ন this_scale = 1/id;
                বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                    যদি (grid_index < 0) {
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            যদি (scale > 0) {
                ভিন্ন id = 1/scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                    }
                    পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                    যদি (grid_index < 0) {
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q2xs + grid_index);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0) scale = sumqx/sumq2;
            }
            যদি (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                uint16_t u = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                যদি (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
                q2[2*ib+1] |= (block_signs[k] << 7*k);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        যদি (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            অনুৱর্ত;
        }

        ভিন্ন d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        ভিন্ন id = 1/d;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/32; ++ib) {
            পূর্ণাঙ্ক l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            q2[2*ib+1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}

স্থির শূন্য quantize_row_iq2_xs_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t n, ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights) {

    ধ্রুৱ পূর্ণাঙ্ক gindex = iq2_data_index(GGML_TYPE_IQ2_XS);

    ধ্রুৱ uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    ধ্রুৱ পূর্ণাঙ্ক      * kmap_q2xs       = iq2_data[gindex].map;
    ধ্রুৱ uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    ধ্রুৱ পূর্ণাঙ্ক kMaxQ = 3;

    ধ্রুৱ int64_t nbl = n/QK_K;

    block_iq2_xs * y = vy;

    ভিন্ন scales[QK_K/16];
    ভিন্ন weight[16];
    ভিন্ন xval[16];
    int8_t L[16];
    int8_t Laux[16];
    ভিন্ন  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2*(QK_K/16)];

    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);
        memset(y[ibl].scales, 0, QK_K/32);

        ভিন্ন max_scale = 0;

        ধ্রুৱ ভিন্ন * xbl = x + QK_K*ibl;
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        ভিন্ন sigma2 = sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {
            ধ্রুৱ ভিন্ন * xb = xbl + 16*ib;
            ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + 16*ib;
            বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                পূর্ণাঙ্ক nflip = 0;
                uint8_t s = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                    যদি (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    অন্যথা {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                যদি (nflip%2) {
                    পূর্ণাঙ্ক imin = 0; ভিন্ন min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    বাবে (পূর্ণাঙ্ক i = 1; i < 8; ++i) {
                        ভিন্ন ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        যদি (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            ভিন্ন max = xval[0];
            বাবে (পূর্ণাঙ্ক i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            যদি (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                অনুৱর্ত;
            }
            ভিন্ন best = 0;
            ভিন্ন scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            বাবে (পূর্ণাঙ্ক is = -9; is <= 9; ++is) {
                ভিন্ন id = (2*kMaxQ-1+is*0.1f)/max;
                ভিন্ন this_scale = 1/id;
                বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    যদি (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) L[i] = Laux[i];
                    বাবে (পূর্ণাঙ্ক k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            পূর্ণাঙ্ক n_not_ongrid = 0;
            বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) যদি (!is_on_grid[k]) ++n_not_ongrid;
            যদি (n_not_ongrid > 0 && scale > 0) {
                ভিন্ন id = 1/scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                    যদি (is_on_grid[k]) অনুৱর্ত;
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                    যদি (grid_index < 0) {
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0) scale = sumqx/sumq2;
            }
            যদি (scale < 0) {
                scale = -scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                uint16_t u = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                যদি (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2*ib+k] = grid_index | (block_signs[k] << 9);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        যদি (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            অনুৱর্ত;
        }

        ভিন্ন d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        ভিন্ন id = 1/d;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {
            পূর্ণাঙ্ক l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            যদি (ib%2 == 0) y[ibl].scales[ib/2] = l;
            অন্যথা y[ibl].scales[ib/2] |= (l << 4);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);

    }
}

size_t quantize_iq2_xxs(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xxs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq2_xxs);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq2_xxs);
}

size_t quantize_iq2_xs(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xs_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq2_xs);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq2_xs);
}

//
// ============================================= 3-bit using D4 lattice
//

প্রকারনির্দেশ সংরচনা {
    uint32_t * grid;
    পূর্ণাঙ্ক      * map;
    uint16_t * neighbours;
} iq3_entry_t;

স্থির iq3_entry_t iq3_data[2] = {
    {রিক্ত, রিক্ত, রিক্ত},
    {রিক্ত, রিক্ত, রিক্ত},
};

স্থির অন্তঃস্থ পূর্ণাঙ্ক iq3_data_index(পূর্ণাঙ্ক grid_size) {
    (শূন্য)grid_size;
    GGML_ASSERT(grid_size == 256 || grid_size == 512);
    ঘূৰাই grid_size == 256 ? 0 : 1;
}

স্থির পূর্ণাঙ্ক iq3_compare_func(ধ্রুৱ শূন্য * বাম, ধ্রুৱ শূন্য * ডান) {
    ধ্রুৱ পূর্ণাঙ্ক * l = (ধ্রুৱ পূর্ণাঙ্ক *)বাম;
    ধ্রুৱ পূর্ণাঙ্ক * r = (ধ্রুৱ পূর্ণাঙ্ক *)ডান;
    ঘূৰাই l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

শূন্য iq3xs_init_impl(পূর্ণাঙ্ক grid_size) {
    ধ্রুৱ পূর্ণাঙ্ক gindex = iq3_data_index(grid_size);
    যদি (iq3_data[gindex].grid) {
        ঘূৰাই;
    }
    স্থির ধ্রুৱ uint16_t kgrid_256[256] = {
            0,     2,     4,     9,    11,    15,    16,    18,    25,    34,    59,    61,    65,    67,    72,    74,
           81,    85,    88,    90,    97,   108,   120,   128,   130,   132,   137,   144,   146,   153,   155,   159,
          169,   175,   189,   193,   199,   200,   202,   213,   248,   267,   287,   292,   303,   315,   317,   321,
          327,   346,   362,   413,   436,   456,   460,   462,   483,   497,   513,   515,   520,   522,   529,   531,
          536,   538,   540,   551,   552,   576,   578,   585,   592,   594,   641,   643,   648,   650,   657,   664,
          698,   704,   706,   720,   729,   742,   758,   769,   773,   808,   848,   852,   870,   889,   901,   978,
          992,  1024,  1026,  1033,  1035,  1040,  1042,  1046,  1049,  1058,  1089,  1091,  1093,  1096,  1098,  1105,
         1112,  1139,  1143,  1144,  1152,  1154,  1161,  1167,  1168,  1170,  1183,  1184,  1197,  1217,  1224,  1228,
         1272,  1276,  1309,  1323,  1347,  1367,  1377,  1404,  1473,  1475,  1486,  1509,  1537,  1544,  1546,  1553,
         1555,  1576,  1589,  1594,  1600,  1602,  1616,  1625,  1636,  1638,  1665,  1667,  1672,  1685,  1706,  1722,
         1737,  1755,  1816,  1831,  1850,  1856,  1862,  1874,  1901,  1932,  1950,  1971,  2011,  2032,  2052,  2063,
         2077,  2079,  2091,  2095,  2172,  2192,  2207,  2208,  2224,  2230,  2247,  2277,  2308,  2345,  2356,  2389,
         2403,  2424,  2501,  2504,  2506,  2520,  2570,  2593,  2616,  2624,  2630,  2646,  2669,  2700,  2714,  2746,
         2754,  2795,  2824,  2835,  2839,  2874,  2882,  2905,  2984,  3028,  3042,  3092,  3108,  3110,  3124,  3153,
         3185,  3215,  3252,  3288,  3294,  3364,  3397,  3434,  3483,  3523,  3537,  3587,  3589,  3591,  3592,  3610,
         3626,  3670,  3680,  3722,  3749,  3754,  3776,  3789,  3803,  3824,  3857,  3873,  3904,  3906,  3924,  3992,
    };
    স্থির ধ্রুৱ uint16_t kgrid_512[512] = {
            0,     1,     2,     5,     7,     8,     9,    10,    12,    14,    16,    17,    21,    27,    32,    34,
           37,    39,    41,    43,    48,    50,    57,    60,    63,    64,    65,    66,    68,    72,    73,    77,
           80,    83,    87,    89,    93,   100,   113,   117,   122,   128,   129,   133,   135,   136,   139,   142,
          145,   149,   152,   156,   162,   165,   167,   169,   171,   184,   187,   195,   201,   205,   208,   210,
          217,   219,   222,   228,   232,   234,   247,   249,   253,   256,   267,   271,   273,   276,   282,   288,
          291,   297,   312,   322,   324,   336,   338,   342,   347,   353,   357,   359,   374,   379,   390,   393,
          395,   409,   426,   441,   448,   450,   452,   464,   466,   470,   475,   488,   492,   512,   513,   514,
          516,   520,   521,   523,   525,   527,   528,   530,   537,   540,   542,   556,   558,   561,   570,   576,
          577,   579,   582,   584,   588,   593,   600,   603,   609,   616,   618,   632,   638,   640,   650,   653,
          655,   656,   660,   666,   672,   675,   685,   688,   698,   705,   708,   711,   712,   715,   721,   727,
          728,   732,   737,   754,   760,   771,   773,   778,   780,   793,   795,   802,   806,   808,   812,   833,
          840,   843,   849,   856,   858,   873,   912,   916,   919,   932,   934,   961,   963,   968,   970,   977,
          989,   993,  1010,  1016,  1024,  1025,  1027,  1029,  1031,  1032,  1034,  1036,  1038,  1041,  1043,  1047,
         1048,  1050,  1057,  1059,  1061,  1064,  1066,  1079,  1080,  1083,  1085,  1088,  1090,  1096,  1099,  1103,
         1106,  1109,  1113,  1116,  1122,  1129,  1153,  1156,  1159,  1169,  1171,  1176,  1183,  1185,  1195,  1199,
         1209,  1212,  1216,  1218,  1221,  1225,  1234,  1236,  1241,  1243,  1250,  1256,  1270,  1281,  1287,  1296,
         1299,  1306,  1309,  1313,  1338,  1341,  1348,  1353,  1362,  1375,  1376,  1387,  1400,  1408,  1410,  1415,
         1425,  1453,  1457,  1477,  1481,  1494,  1496,  1507,  1512,  1538,  1545,  1547,  1549,  1551,  1554,  1561,
         1563,  1565,  1570,  1572,  1575,  1577,  1587,  1593,  1601,  1603,  1605,  1612,  1617,  1619,  1632,  1648,
         1658,  1662,  1664,  1674,  1680,  1690,  1692,  1704,  1729,  1736,  1740,  1745,  1747,  1751,  1752,  1761,
         1763,  1767,  1773,  1787,  1795,  1801,  1806,  1810,  1817,  1834,  1840,  1844,  1857,  1864,  1866,  1877,
         1882,  1892,  1902,  1915,  1934,  1953,  1985,  1987,  2000,  2002,  2013,  2048,  2052,  2058,  2064,  2068,
         2071,  2074,  2081,  2088,  2104,  2114,  2119,  2121,  2123,  2130,  2136,  2141,  2147,  2153,  2157,  2177,
         2179,  2184,  2189,  2193,  2203,  2208,  2223,  2226,  2232,  2244,  2249,  2251,  2256,  2258,  2265,  2269,
         2304,  2306,  2324,  2335,  2336,  2361,  2373,  2375,  2385,  2418,  2443,  2460,  2480,  2504,  2509,  2520,
         2531,  2537,  2562,  2568,  2572,  2578,  2592,  2596,  2599,  2602,  2614,  2620,  2625,  2627,  2629,  2634,
         2641,  2650,  2682,  2688,  2697,  2707,  2712,  2718,  2731,  2754,  2759,  2760,  2775,  2788,  2793,  2805,
         2811,  2817,  2820,  2832,  2842,  2854,  2890,  2902,  2921,  2923,  2978,  3010,  3012,  3026,  3081,  3083,
         3085,  3097,  3099,  3120,  3136,  3152,  3159,  3188,  3210,  3228,  3234,  3245,  3250,  3256,  3264,  3276,
         3281,  3296,  3349,  3363,  3378,  3392,  3395,  3420,  3440,  3461,  3488,  3529,  3531,  3584,  3588,  3591,
         3600,  3602,  3614,  3616,  3628,  3634,  3650,  3657,  3668,  3683,  3685,  3713,  3716,  3720,  3726,  3729,
         3736,  3753,  3778,  3802,  3805,  3819,  3841,  3845,  3851,  3856,  3880,  3922,  3938,  3970,  3993,  4032,
    };

    ধ্রুৱ পূর্ণাঙ্ক kmap_size = 4096;
    ধ্রুৱ পূর্ণাঙ্ক nwant = grid_size == 256 ? 2 : 3;
    ধ্রুৱ uint16_t * kgrid = grid_size == 256 ? kgrid_256 : kgrid_512;
    uint32_t * kgrid_q3xs;
    পূর্ণাঙ্ক      * kmap_q3xs;
    uint16_t * kneighbors_q3xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);
    uint32_t * the_grid = (uint32_t *)malloc(grid_size*পরিমাণ(uint32_t));
    বাবে (পূর্ণাঙ্ক k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
            পূর্ণাঙ্ক l = (kgrid[k] >> 3*i) & 0x7;
            pos[i] = 2*l + 1;
        }
    }
    kgrid_q3xs = the_grid;
    iq3_data[gindex].grid = the_grid;
    kmap_q3xs = (পূর্ণাঙ্ক *)malloc(kmap_size*পরিমাণ(পূর্ণাঙ্ক));
    iq3_data[gindex].map = kmap_q3xs;
    বাবে (পূর্ণাঙ্ক i = 0; i < kmap_size; ++i) kmap_q3xs[i] = -1;
    uint32_t aux32;
    uint8_t * aux8 = (uint8_t *)&aux32;
    বাবে (পূর্ণাঙ্ক i = 0; i < grid_size; ++i) {
        aux32 = kgrid_q3xs[i];
        uint16_t index = 0;
        বাবে (পূর্ণাঙ্ক k=0; k<4; ++k) {
            uint16_t q = (aux8[k] - 1)/2;
            index |= (q << 3*k);
        }
        kmap_q3xs[index] = i;
    }
    // See explanation of parallelism in iq2xs_init_impl
    পূর্ণাঙ্ক * n_per_i = (পূর্ণাঙ্ক *)malloc(kmap_size*পরিমাণ(পূর্ণাঙ্ক));
    GGML_ASSERT(n_per_i);
    পূর্ণাঙ্ক অঙ্ক_neighbors = 0, অঙ্ক_not_in_map = 0;
#ifdef GGML_USE_OPENMP
    #pragma omp parallel reduction(+:অঙ্ক_neighbors,অঙ্ক_not_in_map)
#endif
    {
        পূর্ণাঙ্ক * dist2 = (পূর্ণাঙ্ক *)malloc(2*grid_size*পরিমাণ(পূর্ণাঙ্ক));
        GGML_ASSERT(dist2);
        int8_t pos[4];
        পূর্ণাঙ্ক i;
#ifdef GGML_USE_OPENMP
        #pragma omp বাবে schedule(dynamic, 64)
#endif
        বাবে (i = 0; i < kmap_size; ++i) {
            যদি (kmap_q3xs[i] >= 0) {
                n_per_i[i] = 0;
                অনুৱর্ত;
            }
            ++অঙ্ক_not_in_map;
            বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                পূর্ণাঙ্ক l = (i >> 3*k) & 0x7;
                pos[k] = 2*l + 1;
            }
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q3xs + j);
                পূর্ণাঙ্ক d2 = 0;
                বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
                dist2[2*j+0] = d2;
                dist2[2*j+1] = j;
            }
            qsort(dist2, grid_size, 2*পরিমাণ(পূর্ণাঙ্ক), iq3_compare_func);
            পূর্ণাঙ্ক n = 0; পূর্ণাঙ্ক d2 = dist2[0];
            পূর্ণাঙ্ক nhave = 1;
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                যদি (dist2[2*j] > d2) {
                    যদি (nhave == nwant) ভঙ্গ;
                    d2 = dist2[2*j];
                    ++nhave;
                }
                ++n;
            }
            n_per_i[i] = n;
            অঙ্ক_neighbors += n;
        }
        মুক্ত(dist2);
    }
    //printf("%s: %d neighbours in total\n", __func__, num_neighbors);
    kneighbors_q3xs = (uint16_t *)malloc((অঙ্ক_neighbors + অঙ্ক_not_in_map)*পরিমাণ(uint16_t));
    iq3_data[gindex].neighbours = kneighbors_q3xs;

    পূর্ণাঙ্ক * offsets = (পূর্ণাঙ্ক *)malloc(kmap_size*পরিমাণ(পূর্ণাঙ্ক));
    GGML_ASSERT(offsets);
    পূর্ণাঙ্ক counter = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < kmap_size; ++i) {
        যদি (kmap_q3xs[i] >= 0) {
            offsets[i] = -1;
            অনুৱর্ত;
        }
        offsets[i] = counter;
        counter += 1 + n_per_i[i];
    }

#ifdef GGML_USE_OPENMP
    #pragma omp parallel
#endif
    {
        পূর্ণাঙ্ক * dist2 = (পূর্ণাঙ্ক *)malloc(2*grid_size*পরিমাণ(পূর্ণাঙ্ক));
        GGML_ASSERT(dist2);
        int8_t pos[4];
        পূর্ণাঙ্ক i;
#ifdef GGML_USE_OPENMP
        #pragma omp বাবে schedule(dynamic, 64)
#endif
        বাবে (i = 0; i < kmap_size; ++i) {
            যদি (kmap_q3xs[i] >= 0) অনুৱর্ত;
            বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                পূর্ণাঙ্ক l = (i >> 3*k) & 0x7;
                pos[k] = 2*l + 1;
            }
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q3xs + j);
                পূর্ণাঙ্ক d2 = 0;
                বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
                dist2[2*j+0] = d2;
                dist2[2*j+1] = j;
            }
            qsort(dist2, grid_size, 2*পরিমাণ(পূর্ণাঙ্ক), iq3_compare_func);
            পূর্ণাঙ্ক local_counter = offsets[i];
            kmap_q3xs[i] = -(local_counter + 1);
            পূর্ণাঙ্ক d2 = dist2[0];
            uint16_t * start = &kneighbors_q3xs[local_counter++];
            পূর্ণাঙ্ক n = 0, nhave = 1;
            বাবে (পূর্ণাঙ্ক j = 0; j < grid_size; ++j) {
                যদি (dist2[2*j] > d2) {
                    যদি (nhave == nwant) ভঙ্গ;
                    d2 = dist2[2*j];
                    ++nhave;
                }
                kneighbors_q3xs[local_counter++] = dist2[2*j+1];
                ++n;
            }
            *start = n;
        }
        মুক্ত(dist2);
    }
    মুক্ত(offsets);
    মুক্ত(n_per_i);
}

শূন্য iq3xs_free_impl(পূর্ণাঙ্ক grid_size) {
    GGML_ASSERT(grid_size == 256 || grid_size == 512);
    ধ্রুৱ পূর্ণাঙ্ক gindex = iq3_data_index(grid_size);
    যদি (iq3_data[gindex].grid) {
        মুক্ত(iq3_data[gindex].grid);       iq3_data[gindex].grid = রিক্ত;
        মুক্ত(iq3_data[gindex].map);        iq3_data[gindex].map  = রিক্ত;
        মুক্ত(iq3_data[gindex].neighbours); iq3_data[gindex].neighbours = রিক্ত;
    }
}

স্থির পূর্ণাঙ্ক iq3_find_best_neighbour(ধ্রুৱ uint16_t * GGML_RESTRICT neighbours, ধ্রুৱ uint32_t * GGML_RESTRICT grid,
        ধ্রুৱ ভিন্ন * GGML_RESTRICT xval, ধ্রুৱ ভিন্ন * GGML_RESTRICT weight, ভিন্ন scale, int8_t * GGML_RESTRICT L) {
    পূর্ণাঙ্ক অঙ্ক_neighbors = neighbours[0];
    GGML_ASSERT(অঙ্ক_neighbors > 0);
    ভিন্ন best_d2 = FLT_MAX;
    পূর্ণাঙ্ক grid_index = -1;
    বাবে (পূর্ণাঙ্ক j = 1; j <= অঙ্ক_neighbors; ++j) {
        ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + neighbours[j]);
        ভিন্ন d2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
            ভিন্ন q = pg[i];
            ভিন্ন diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        যদি (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + grid_index);
    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    ঘূৰাই grid_index;
}

স্থির শূন্য quantize_row_iq3_xxs_impl(পূর্ণাঙ্ক grid_size, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t n,
        ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights) {

    ধ্রুৱ পূর্ণাঙ্ক gindex = iq3_data_index(grid_size);

    ধ্রুৱ uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    ধ্রুৱ পূর্ণাঙ্ক      * kmap_q3xs       = iq3_data[gindex].map;
    ধ্রুৱ uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    ধ্রুৱ পূর্ণাঙ্ক kMaxQ = 8;

    ধ্রুৱ int64_t nbl = n/QK_K;

    ggml_fp16_t * dh;
    uint8_t * qs;
    পূর্ণাঙ্ক block_size;
    যদি (grid_size == 256) {
        block_iq3_xxs * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = পরিমাণ(block_iq3_xxs);
    } অন্যথা {
        block_iq3_s * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = পরিমাণ(block_iq3_s);
    }
    পূর্ণাঙ্ক quant_size = block_size - পরিমাণ(ggml_fp16_t);

    ভিন্ন scales[QK_K/32];
    ভিন্ন weight[32];
    ভিন্ন xval[32];
    int8_t L[32];
    int8_t Laux[32];
    ভিন্ন  waux[32];
    bool   is_on_grid[8];
    bool   is_on_grid_aux[8];
    uint8_t block_signs[8];
    uint8_t q3[3*(QK_K/8)+QK_K/32];
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4);
    uint8_t  * qh = q3 + 3*(QK_K/8);

    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nbl; ++ibl) {

        dh[0] = GGML_FP32_TO_FP16(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        ভিন্ন max_scale = 0;

        ধ্রুৱ ভিন্ন * xbl = x + QK_K*ibl;
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        ভিন্ন sigma2 = 2*sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/32; ++ib) {
            ধ্রুৱ ভিন্ন * xb = xbl + 32*ib;
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + 32*ib;
                বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) weight[i] = xb[i]*xb[i];
            }
            বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                পূর্ণাঙ্ক nflip = 0;
                uint8_t s = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                    যদি (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    অন্যথা {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                যদি (nflip%2) {
                    পূর্ণাঙ্ক imin = 0; ভিন্ন min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    বাবে (পূর্ণাঙ্ক i = 1; i < 8; ++i) {
                        ভিন্ন ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        যদি (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            ভিন্ন max = xval[0];
            বাবে (পূর্ণাঙ্ক i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 32);
            যদি (max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
                অনুৱর্ত;
            }
            ভিন্ন best = 0;
            ভিন্ন scale = max/(2*kMaxQ-1);
            বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) is_on_grid[k] = true;
            বাবে (পূর্ণাঙ্ক is = -15; is <= 15; ++is) {
                ভিন্ন id = (2*kMaxQ-1+is*0.2f)/max;
                ভিন্ন this_scale = 1/id;
                বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) {
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    পূর্ণাঙ্ক grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    যদি (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) L[i] = Laux[i];
                    বাবে (পূর্ণাঙ্ক k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            পূর্ণাঙ্ক n_not_ongrid = 0;
            বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) যদি (!is_on_grid[k]) ++n_not_ongrid;
            যদি (n_not_ongrid > 0 && scale > 0) {
                ভিন্ন id = 1/scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) {
                    যদি (is_on_grid[k]) অনুৱর্ত;
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    পূর্ণাঙ্ক grid_index = kmap_q3xs[u];
                    যদি (grid_index < 0) {
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q3xs + grid_index);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 32; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0) scale = sumqx/sumq2;
            }
            যদি (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            বাবে (পূর্ণাঙ্ক k = 0; k < 8; ++k) {
                uint16_t u = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                পূর্ণাঙ্ক grid_index = kmap_q3xs[u];
                যদি (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                যদি (grid_size == 256) {
                    q3[8*ib+k] = grid_index;
                } অন্যথা {
                    q3[8*ib+k] = grid_index & 255;
                    qh[ib] |= ((grid_index >> 8) << k);
                }

            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        যদি (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size/পরিমাণ(ggml_fp16_t);
            qs += block_size;
            অনুৱর্ত;
        }

        ভিন্ন d = max_scale/31;
        dh[0] = GGML_FP32_TO_FP16(d * 1.0125f);  // small improvement via this fudge factor
        ভিন্ন id = 1/d;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/32; ++ib) {
            পূর্ণাঙ্ক l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(qs, q3, quant_size);

        dh += block_size/পরিমাণ(ggml_fp16_t);
        qs += block_size;

    }
}

size_t quantize_iq3_xxs(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_xxs_impl(256, src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq3_xxs);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq3_xxs);
}

শূন্য quantize_row_iq3_xxs_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    quantize_row_iq3_xxs_impl(256, x, y, k, রিক্ত);
}

স্থির শূন্য quantize_row_iq3_s_impl(পূর্ণাঙ্ক block_size, ধ্রুৱ ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, পূর্ণাঙ্ক n,
        ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights,
        ভিন্ন   * scales,
        ভিন্ন   * weight,
        ভিন্ন   * xval,
        int8_t  * L,
        int8_t  * Laux,
        ভিন্ন   * waux,
        bool    * is_on_grid,
        bool    * is_on_grid_aux,
        uint8_t * block_signs) {

    ধ্রুৱ পূর্ণাঙ্ক gindex = iq3_data_index(512);

    ধ্রুৱ uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    ধ্রুৱ পূর্ণাঙ্ক      * kmap_q3xs       = iq3_data[gindex].map;
    ধ্রুৱ uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    ধ্রুৱ পূর্ণাঙ্ক kMaxQ = 8;

    ধ্রুৱ int64_t nbl = n/QK_K;

    block_iq3_s * y = vy;

    ধ্রুৱ পূর্ণাঙ্ক bs4 = block_size/4;
    ধ্রুৱ পূর্ণাঙ্ক bs8 = block_size/8;

    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, পরিমাণ(block_iq3_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        ভিন্ন max_scale = 0;

        ধ্রুৱ ভিন্ন * xbl = x + QK_K*ibl;
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        ভিন্ন sigma2 = 2*sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/block_size; ++ib) {
            ধ্রুৱ ভিন্ন * xb = xbl + block_size*ib;
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + block_size*ib;
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
            বাবে (পূর্ণাঙ্ক k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                    যদি (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    অন্যথা {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            ভিন্ন max = xval[0];
            বাবে (পূর্ণাঙ্ক i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
            memset(L, 0, block_size);
            যদি (!max) {
                scales[ib] = 0;
                অনুৱর্ত;
            }
            ভিন্ন best = 0;
            ভিন্ন scale = max/(2*kMaxQ-1);
            বাবে (পূর্ণাঙ্ক k = 0; k < bs4; ++k) is_on_grid[k] = false;
            বাবে (পূর্ণাঙ্ক is = -9; is <= 9; ++is) {
                ভিন্ন id = (2*kMaxQ-1+is*0.2f)/max;
                ভিন্ন this_scale = 1/id;
                বাবে (পূর্ণাঙ্ক k = 0; k < bs4; ++k) {
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    পূর্ণাঙ্ক grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    যদি (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) L[i] = Laux[i];
                    বাবে (পূর্ণাঙ্ক k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            পূর্ণাঙ্ক n_not_ongrid = 0;
            বাবে (পূর্ণাঙ্ক k = 0; k < bs4; ++k) যদি (!is_on_grid[k]) ++n_not_ongrid;
            যদি (n_not_ongrid > 0 && scale > 0) {
                ভিন্ন id = 1/scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < bs4; ++k) {
                    //if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    পূর্ণাঙ্ক grid_index = kmap_q3xs[u];
                    যদি (grid_index < 0) {
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q3xs + grid_index);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0) scale = sumqx/sumq2;
            }
            যদি (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            বাবে (পূর্ণাঙ্ক k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                পূর্ণাঙ্ক grid_index = kmap_q3xs[u];
                যদি (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            বাবে (পূর্ণাঙ্ক k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        যদি (!max_scale) {
            অনুৱর্ত;
        }

        ভিন্ন d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 1.033f);
        ভিন্ন id = 1/d;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/block_size; ib += 2) {
            পূর্ণাঙ্ক l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
            l1 = MAX(0, MIN(15, l1));
            পূর্ণাঙ্ক l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
            l2 = MAX(0, MIN(15, l2));
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }

    }
}

#define IQ3S_BLOCK_SIZE 32
size_t quantize_iq3_s(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    ভিন্ন scales[QK_K/IQ3S_BLOCK_SIZE];
    ভিন্ন weight[IQ3S_BLOCK_SIZE];
    ভিন্ন xval[IQ3S_BLOCK_SIZE];
    int8_t L[IQ3S_BLOCK_SIZE];
    int8_t Laux[IQ3S_BLOCK_SIZE];
    ভিন্ন  waux[IQ3S_BLOCK_SIZE];
    bool   is_on_grid[IQ3S_BLOCK_SIZE/4];
    bool   is_on_grid_aux[IQ3S_BLOCK_SIZE/4];
    uint8_t block_signs[IQ3S_BLOCK_SIZE/8];
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_s_impl(IQ3S_BLOCK_SIZE, src, qrow, n_per_row, quant_weights,
                scales, weight, xval, L, Laux, waux, is_on_grid, is_on_grid_aux, block_signs);
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq3_s);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq3_s);
}

শূন্য quantize_row_iq3_s_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq3_s * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    quantize_iq3_s(x, y, 1, k, রিক্ত);
}


// =================================== 1.5 bpw ===================================================

স্থির পূর্ণাঙ্ক iq1_find_best_neighbour(ধ্রুৱ uint16_t * GGML_RESTRICT neighbours, ধ্রুৱ uint64_t * GGML_RESTRICT grid,
        ধ্রুৱ ভিন্ন * GGML_RESTRICT xval, ধ্রুৱ ভিন্ন * GGML_RESTRICT weight, ভিন্ন * scale, int8_t * GGML_RESTRICT L, পূর্ণাঙ্ক ngrid) {
    পূর্ণাঙ্ক অঙ্ক_neighbors = neighbours[0];
    GGML_ASSERT(অঙ্ক_neighbors > 0);
    ভিন্ন best_score = -FLT_MAX;
    পূর্ণাঙ্ক grid_index = -1;
    বাবে (পূর্ণাঙ্ক j = 1; j <= অঙ্ক_neighbors; ++j) {
        ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + neighbours[j]);
        ভিন্ন sumqx = 0, sumq2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
            ভিন্ন q = (pg[i] - 3)/2;
            ভিন্ন w = weight[i];
            sumqx += w*q*xval[i];
            sumq2 += w*q*q;
        }
        যদি (sumqx > 0 && sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
            *scale = sumqx/sumq2; best_score = *scale * sumqx;
            grid_index = neighbours[j];
        }
    }
    যদি (grid_index < 0) {
        বাবে (পূর্ণাঙ্ক i = 0; i < ngrid; ++i) {
            ধ্রুৱ int8_t * grid_i = (ধ্রুৱ int8_t *)(grid + i);
            ভিন্ন sumqx = 0, sumq2 = 0;
            বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                ভিন্ন w = weight[j];
                ভিন্ন q = (grid_i[j] - 3)/2;
                sumqx += w*q*xval[j];
                sumq2 += w*q*q;
            }
            যদি (sumqx > 0 && sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                *scale = sumqx/sumq2; best_score = *scale*sumqx;
                grid_index = i;
            }
        }
    }
    যদি (grid_index < 0) {
        printf("Oops, did not find grid point\n");
        printf("Have %d neighbours\n", অঙ্ক_neighbors);
        বাবে (পূর্ণাঙ্ক j = 1; j <= অঙ্ক_neighbors; ++j) {
            ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + neighbours[j]);
            ভিন্ন sumqx = 0, sumq2 = 0;
            বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                ভিন্ন q = (pg[i] - 3)/2;
                ভিন্ন w = weight[i];
                sumqx += w*q*xval[i];
                sumq2 += w*q*q;
            }
            printf("    neighbour %d: sumqx = %g sumq2 = %g\n", j, (দ্ৱিগুণ)sumqx, (দ্ৱিগুণ)sumq2);
        }
    }
    GGML_ASSERT(grid_index >= 0);
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    *scale *= 1.05f;  // This is a fudge factor. Don't ask me why it improves the result.
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + grid_index);
    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    ঘূৰাই grid_index;
}

স্থির পূর্ণাঙ্ক iq1_find_best_neighbour2(ধ্রুৱ uint16_t * GGML_RESTRICT neighbours, ধ্রুৱ uint64_t * GGML_RESTRICT grid,
        ধ্রুৱ ভিন্ন * GGML_RESTRICT xval, ধ্রুৱ ভিন্ন * GGML_RESTRICT weight, ভিন্ন scale, ধ্রুৱ ভিন্ন * GGML_RESTRICT xg, int8_t * GGML_RESTRICT L, পূর্ণাঙ্ক ngrid) {
    পূর্ণাঙ্ক অঙ্ক_neighbors = neighbours[0];
    GGML_ASSERT(অঙ্ক_neighbors > 0);
    ভিন্ন best_score = FLT_MAX;
    পূর্ণাঙ্ক grid_index = -1;
    বাবে (পূর্ণাঙ্ক j = 1; j <= অঙ্ক_neighbors; ++j) {
        ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + neighbours[j]);
        ভিন্ন d2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
            ভিন্ন q = xg[(pg[i] - 1)/2];
            ভিন্ন w = weight[i];
            ভিন্ন diff = scale*q - xval[i];
            d2 += w*diff*diff;
        }
        যদি (d2 < best_score) {
            best_score = d2;
            grid_index = neighbours[j];
        }
    }
    যদি (grid_index < 0) {
        বাবে (পূর্ণাঙ্ক i = 0; i < ngrid; ++i) {
            ধ্রুৱ int8_t * grid_i = (ধ্রুৱ int8_t *)(grid + i);
            ভিন্ন d2 = 0;
            বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                ভিন্ন w = weight[j];
                ভিন্ন q = xg[(grid_i[j] - 1)/2];
                ভিন্ন diff = scale*q - xval[i];
                d2 += w*diff*diff;
            }
            যদি (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    যদি (grid_index < 0) {
        printf("Oops, did not find grid point\n");
        printf("Have %d neighbours\n", অঙ্ক_neighbors);
        বাবে (পূর্ণাঙ্ক j = 1; j <= অঙ্ক_neighbors; ++j) {
            ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + neighbours[j]);
            ভিন্ন sumqx = 0, sumq2 = 0;
            বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                ভিন্ন q = xg[(pg[i] - 1)/2];
                ভিন্ন w = weight[i];
                sumqx += w*q*xval[i];
                sumq2 += w*q*q;
            }
            printf("    neighbour %d: sumqx = %g sumq2 = %g\n", j, (দ্ৱিগুণ)sumqx, (দ্ৱিগুণ)sumq2);
        }
    }
    GGML_ASSERT(grid_index >= 0);
    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(grid + grid_index);
    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    ঘূৰাই grid_index;
}

স্থির পূর্ণাঙ্ক iq1_sort_helper(ধ্রুৱ শূন্য * বাম, ধ্রুৱ শূন্য * ডান) {
    ধ্রুৱ ভিন্ন * l = বাম;
    ধ্রুৱ ভিন্ন * r = ডান;
    ঘূৰাই *l < *r ? -1 : *l > *r ? 1 : 0;
}

#define IQ1S_BLOCK_SIZE 32
#define IQ1M_BLOCK_SIZE 16
স্থির শূন্য quantize_row_iq1_s_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t n, ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights,
        ভিন্ন    * scales,
        ভিন্ন    * weight,
        ভিন্ন    * sumx,
        ভিন্ন    * sumw,
        ভিন্ন    * pairs,
        int8_t   * L,
        uint16_t * index,
        int8_t   * shifts) {

    ধ্রুৱ পূর্ণাঙ্ক gindex = iq2_data_index(GGML_TYPE_IQ1_S);

    ধ্রুৱ uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    ধ্রুৱ পূর্ণাঙ্ক      * kmap_q2xs       = iq2_data[gindex].map;
    ধ্রুৱ uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    block_iq1_s * y = vy;

    ধ্রুৱ int64_t nbl = n/QK_K;

    ধ্রুৱ পূর্ণাঙ্ক block_size = IQ1S_BLOCK_SIZE;

    ধ্রুৱ ভিন্ন x_p[3] = {-1 + IQ1S_DELTA,  IQ1S_DELTA, 1 + IQ1S_DELTA};
    ধ্রুৱ ভিন্ন x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};


    পূর্ণাঙ্ক * idx = (পূর্ণাঙ্ক *)(pairs + 1);

    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);

        ভিন্ন max_scale = 0;

        ধ্রুৱ ভিন্ন * xbl = x + QK_K*ibl;
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        ভিন্ন sigma2 = 2*sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/block_size; ++ib) {
            ধ্রুৱ ভিন্ন * xb = xbl + block_size*ib;
            ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + block_size*ib;
            বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            ভিন্ন max = fabsf(xb[0]);
            বাবে (পূর্ণাঙ্ক i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
            যদি (max < GROUP_MAX_EPS_IQ1_S) {
                scales[ib] = 0;
                shifts[ib] = 1;
                memset(L, 1, block_size);
                অনুৱর্ত;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            qsort(pairs, block_size, 2*পরিমাণ(ভিন্ন), iq1_sort_helper);
            {
                sumx[0] = sumw[0] = 0;
                বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
                    পূর্ণাঙ্ক i = idx[2*j];
                    sumx[j+1] = sumx[j] + weight[i]*xb[i];
                    sumw[j+1] = sumw[j] + weight[i];
                }
            }
            ভিন্ন best_score = -FLT_MAX, scale = max;
            পূর্ণাঙ্ক besti1 = -1, besti2 = -1, best_shift = 0;
            বাবে (পূর্ণাঙ্ক i1 = 0; i1 <= block_size; ++i1) {
                বাবে (পূর্ণাঙ্ক i2 = i1; i2 <= block_size; ++i2) {
                    ভিন্ন sumqx = (sumx[i1] - sumx[0])*x_p[0] + (sumx[i2] - sumx[i1])*x_p[1] + (sumx[block_size] - sumx[i2])*x_p[2];
                    ভিন্ন sumq2 = (sumw[i1] - sumw[0])*x_p[0]*x_p[0] + (sumw[i2] - sumw[i1])*x_p[1]*x_p[1] + (sumw[block_size] - sumw[i2])*x_p[2]*x_p[2];
                    যদি (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                        scale = sumqx/sumq2; best_score = scale*sumqx;
                        besti1 = i1; besti2 = i2; best_shift = 1;
                    }
                    sumqx = (sumx[i1] - sumx[0])*x_m[0] + (sumx[i2] - sumx[i1])*x_m[1] + (sumx[block_size] - sumx[i2])*x_m[2];
                    sumq2 = (sumw[i1] - sumw[0])*x_m[0]*x_m[0] + (sumw[i2] - sumw[i1])*x_m[1]*x_m[1] + (sumw[block_size] - sumw[i2])*x_m[2]*x_m[2];
                    যদি (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                        scale = sumqx/sumq2; best_score = scale*sumqx;
                        besti1 = i1; besti2 = i2; best_shift = -1;
                    }
                }
            }
            যদি (besti1 < 0 || besti2 < 0 || best_shift == 0) {
                scales[ib] = 0;
                shifts[ib] = 1;
                memset(L, 1, block_size);
                অনুৱর্ত;
            }
            বাবে (পূর্ণাঙ্ক j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            বাবে (পূর্ণাঙ্ক j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            বাবে (পূর্ণাঙ্ক j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            যদি (scale < 0) {
                বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale; best_shift = -best_shift;
            }
            bool all_on_grid = true;
            ধ্রুৱ ভিন্ন * xx = best_shift == 1 ? x_p : x_m;
            বাবে (পূর্ণাঙ্ক k = 0; k < block_size/8; ++k) {
                uint16_t u = 0;
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
                পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                যদি (grid_index < 0) {
                    all_on_grid = false;
                    ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
                    GGML_ASSERT(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            যদি (!all_on_grid) {
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক k = 0; k < block_size/8; ++k) {
                    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q2xs + index[k]);
                    বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                        ভিন্ন w = weight[8*k + j];
                        ভিন্ন q = xx[(pg[j] - 1)/2];
                        sumqx += w*q*xb[8*k+j];
                        sumq2 += w*q*q;
                    }
                }
                যদি (sumqx > 0 && sumq2 > 0) scale = sumqx/sumq2;
            }
            uint16_t h = 0;
            বাবে (পূর্ণাঙ্ক k = 0; k < block_size/8; ++k) {
                y[ibl].qs[(block_size/8)*ib + k] = index[k] & 255;
                h |= (index[k] >> 8) << 3*k;
            }
            y[ibl].qh[ib] = h;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = MAX(max_scale, scale);
        }

        যদি (!max_scale) {
            অনুৱর্ত;
        }

        ভিন্ন d = max_scale/15;
        y[ibl].d = GGML_FP32_TO_FP16(d*1.125f); // 1.125f is another fudge factor. Don't ask me why it is needed.
        ভিন্ন id = 1/d;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/block_size; ++ib) {
            পূর্ণাঙ্ক l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(7, l));
            যদি (shifts[ib] == -1) l |= 8;
            y[ibl].qh[ib] |= (l << 12);
        }
    }
}

size_t quantize_iq1_s(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    ভিন্ন  scales[QK_K/IQ1S_BLOCK_SIZE];
    ভিন্ন  weight[IQ1S_BLOCK_SIZE];
    int8_t L[IQ1S_BLOCK_SIZE];
    ভিন্ন  sumx[IQ1S_BLOCK_SIZE+1];
    ভিন্ন  sumw[IQ1S_BLOCK_SIZE+1];
    ভিন্ন  pairs[2*IQ1S_BLOCK_SIZE];
    uint16_t index[IQ1S_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1S_BLOCK_SIZE];
    int64_t nblock = n_per_row/QK_K;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq1_s_impl(src, qrow, n_per_row, quant_weights, scales, weight, sumx, sumw, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq1_s);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq1_s);
}

স্থির শূন্য quantize_row_iq1_m_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t n, ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights,
        ভিন্ন    * scales,
        ভিন্ন    * weight,
        ভিন্ন    * pairs,
        int8_t   * L,
        uint16_t * index,
        int8_t   * shifts) {

    ধ্রুৱ পূর্ণাঙ্ক gindex = iq2_data_index(GGML_TYPE_IQ1_M);

    ধ্রুৱ uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    ধ্রুৱ পূর্ণাঙ্ক      * kmap_q2xs       = iq2_data[gindex].map;
    ধ্রুৱ uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    block_iq1_m * y = vy;

    ধ্রুৱ int64_t nbl = n/QK_K;

    ধ্রুৱ পূর্ণাঙ্ক block_size = IQ1M_BLOCK_SIZE;

    ধ্রুৱ ভিন্ন x_p[3] = {-1 + IQ1M_DELTA,  IQ1M_DELTA, 1 + IQ1M_DELTA};
    ধ্রুৱ ভিন্ন x_m[3] = {-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA};
    ধ্রুৱ uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};

    পূর্ণাঙ্ক * idx = (পূর্ণাঙ্ক *)(pairs + 1);

    ভিন্ন sumqx[4], sumq2[4];

    iq1m_scale_t s;
    ধ্রুৱ ভিন্ন * xx;

    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nbl; ++ibl) {
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);
        memset(y[ibl].scales, 0, QK_K/32);

        ভিন্ন max_scale = 0;

        ধ্রুৱ ভিন্ন * xbl = x + QK_K*ibl;
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        ভিন্ন sigma2 = 2*sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/block_size; ++ib) {
            ধ্রুৱ ভিন্ন * xb = xbl + block_size*ib;
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + block_size*ib;
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            ভিন্ন max = fabsf(xb[0]);
            বাবে (পূর্ণাঙ্ক i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
            যদি (max < GROUP_MAX_EPS_IQ1_M) {
                scales[ib] = 0;
                shifts[ib] = 0;
                memset(L, 1, block_size);
                অনুৱর্ত;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            qsort(pairs, block_size, 2*পরিমাণ(ভিন্ন), iq1_sort_helper);
            ভিন্ন best_score = -FLT_MAX, scale = max;
            পূর্ণাঙ্ক besti1 = -1, besti2 = -1, best_k = -1;
            // 0: +, +
            // 1: +, -
            // 2: -, +
            // 3: -, -
            // prefix sums are kept per half of the block because each half can use a different sign (x_p or x_m)
            // since v[0]-v[1] = v[1]-v[2] = -1 for both x_p and x_m, the 3-group sum for a split collapses to T*v[2] - px[i1] - px[i2]
            {
                ভিন্ন px[2][IQ1M_BLOCK_SIZE+1];
                ভিন্ন pw[2][IQ1M_BLOCK_SIZE+1];
                px[0][0] = px[1][0] = 0;
                pw[0][0] = pw[1][0] = 0;
                বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
                    ধ্রুৱ পূর্ণাঙ্ক i = idx[2*j];
                    ধ্রুৱ পূর্ণাঙ্ক h = i < block_size/2 ? 0 : 1;
                    px[h][j+1]   = px[h][j] + weight[i]*xb[i];
                    px[1-h][j+1] = px[1-h][j];
                    pw[h][j+1]   = pw[h][j] + weight[i];
                    pw[1-h][j+1] = pw[1-h][j];
                }
                ধ্রুৱ ভিন্ন txs[2] = {px[0][block_size], px[1][block_size]}; // total weight*x per half
                ধ্রুৱ ভিন্ন tws[2] = {pw[0][block_size], pw[1][block_size]}; // total weight per half
                ধ্রুৱ ভিন্ন p2   = x_p[2], m2 = x_m[2];
                ধ্রুৱ ভিন্ন cp1  = x_p[0]*x_p[0] - x_p[1]*x_p[1];
                ধ্রুৱ ভিন্ন cp2  = x_p[1]*x_p[1] - x_p[2]*x_p[2];
                ধ্রুৱ ভিন্ন cm1  = x_m[0]*x_m[0] - x_m[1]*x_m[1];
                ধ্রুৱ ভিন্ন cm2  = x_m[1]*x_m[1] - x_m[2]*x_m[2];
                বাবে (পূর্ণাঙ্ক i1 = 0; i1 <= block_size; ++i1) {
                    বাবে (পূর্ণাঙ্ক i2 = i1; i2 <= block_size; ++i2) {
                        ভিন্ন qx_p[2], qx_m[2], q2_p[2], q2_m[2];
                        বাবে (পূর্ণাঙ্ক h = 0; h < 2; ++h) {
                            ধ্রুৱ ভিন্ন sx = px[h][i1] + px[h][i2];
                            qx_p[h] = txs[h]*p2 - sx;
                            qx_m[h] = txs[h]*m2 - sx;
                            q2_p[h] = tws[h]*p2*p2 + pw[h][i1]*cp1 + pw[h][i2]*cp2;
                            q2_m[h] = tws[h]*m2*m2 + pw[h][i1]*cm1 + pw[h][i2]*cm2;
                        }
                        sumqx[0] = qx_p[0] + qx_p[1];
                        sumqx[1] = qx_p[0] + qx_m[1];
                        sumqx[2] = qx_m[0] + qx_p[1];
                        sumqx[3] = qx_m[0] + qx_m[1];
                        sumq2[0] = q2_p[0] + q2_p[1];
                        sumq2[1] = q2_p[0] + q2_m[1];
                        sumq2[2] = q2_m[0] + q2_p[1];
                        sumq2[3] = q2_m[0] + q2_m[1];
                        বাবে (পূর্ণাঙ্ক k = 0; k < 4; ++k) {
                            যদি (sumq2[k] > 0 && sumqx[k]*sumqx[k] > best_score*sumq2[k]) {
                                scale = sumqx[k]/sumq2[k]; best_score = scale*sumqx[k];
                                besti1 = i1; besti2 = i2; best_k = k;
                            }
                        }
                    }
                }
            }
            যদি (besti1 < 0 || besti2 < 0 || best_k < 0) {
                scales[ib] = 0;
                shifts[ib] = 0;
                memset(L, 1, block_size);
                অনুৱর্ত;
            }
            বাবে (পূর্ণাঙ্ক j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            বাবে (পূর্ণাঙ্ক j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            বাবে (পূর্ণাঙ্ক j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            যদি (scale < 0) {
                বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale;
                best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
            }
            bool all_on_grid = true;
            বাবে (পূর্ণাঙ্ক k = 0; k < block_size/8; ++k) {
                যদি (k == 0) xx = best_k < 2 ? x_p : x_m;
                অন্যথা xx = best_k%2 == 0 ? x_p : x_m;
                uint16_t u = 0;
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
                পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                যদি (grid_index < 0) {
                    all_on_grid = false;
                    ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
                    GGML_ASSERT(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            যদি (!all_on_grid) {
                ভিন্ন sumqx_f = 0, sumq2_f = 0;
                বাবে (পূর্ণাঙ্ক k = 0; k < block_size/8; ++k) {
                    যদি (k == 0) xx = best_k < 2 ? x_p : x_m;
                    অন্যথা xx = best_k%2 == 0 ? x_p : x_m;
                    ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q2xs + index[k]);
                    বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                        ভিন্ন w = weight[8*k + j];
                        ভিন্ন q = xx[(pg[j] - 1)/2];
                        sumqx_f += w*q*xb[8*k+j];
                        sumq2_f += w*q*q;
                    }
                }
                যদি (sumqx_f > 0 && sumq2_f > 0) scale = sumqx_f/sumq2_f;
            }
            y[ibl].qs[2*ib + 0] = index[0] & 255;
            y[ibl].qs[2*ib + 1] = index[1] & 255;
            y[ibl].qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_k;
            max_scale = MAX(max_scale, scale);
        }

        যদি (!max_scale) {
            অনুৱর্ত;
        }

        uint16_t * sc = (uint16_t *)y[ibl].scales;
        ভিন্ন d = max_scale/15;
        ভিন্ন id = 1/d;
        ভিন্ন sumqx_f = 0, sumq2_f = 0;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/block_size; ++ib) {
            পূর্ণাঙ্ক l = nearest_int(0.5f*(id*scales[ib+0]-1));
            l = MAX(0, MIN(7, l));
            sc[ib/4] |= (l << 3*(ib%4));
            y[ibl].qh[ib] |= masks[shifts[ib]];
            ধ্রুৱ ভিন্ন * xb = xbl + block_size*ib;
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + block_size*ib;
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            বাবে (পূর্ণাঙ্ক k = 0; k < block_size/8; ++k) {
                যদি (k == 0) xx = shifts[ib] < 2 ? x_p : x_m;
                অন্যথা xx = shifts[ib]%2 == 0 ? x_p : x_m;
                ধ্রুৱ int8_t * pg = (ধ্রুৱ int8_t *)(kgrid_q2xs + y[ibl].qs[2*ib+k] + ((y[ibl].qh[ib] << (8 - 4*k)) & 0x700));
                বাবে (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                    ভিন্ন w = weight[8*k + j];
                    ভিন্ন q = xx[(pg[j] - 1)/2]*(2*l+1);
                    sumqx_f += w*q*xb[8*k+j];
                    sumq2_f += w*q*q;
                }
            }
        }
        যদি (sumq2_f > 0) d = sumqx_f/sumq2_f;
        s.f16 = GGML_FP32_TO_FP16(d*1.1125f); // 1.1125f is another fudge factor. Don't ask me why it is needed.
        sc[0] |= ((s.u16 & 0x000f) << 12);
        sc[1] |= ((s.u16 & 0x00f0) <<  8);
        sc[2] |= ((s.u16 & 0x0f00) <<  4);
        sc[3] |= ((s.u16 & 0xf000) <<  0);
    }
}

size_t quantize_iq1_m(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    ভিন্ন  scales[QK_K/IQ1M_BLOCK_SIZE];
    ভিন্ন  weight[IQ1M_BLOCK_SIZE];
    int8_t L[IQ1M_BLOCK_SIZE];
    ভিন্ন  pairs[2*IQ1M_BLOCK_SIZE];
    uint16_t index[IQ1M_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1M_BLOCK_SIZE];
    int64_t nblock = n_per_row/QK_K;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq1_m_impl(src, qrow, n_per_row, quant_weights, scales, weight, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq1_m);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq1_m);
}

// ============================ 4-bit non-linear quants

স্থির শূন্য quantize_row_iq4_nl_impl(ধ্রুৱ পূর্ণাঙ্ক super_block_size, ধ্রুৱ পূর্ণাঙ্ক block_size, ধ্রুৱ ভিন্ন * GGML_RESTRICT x,
        ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        ভিন্ন * scales, ভিন্ন * weight, uint8_t * L,
        ধ্রুৱ int8_t * মানসমূহ,
        ধ্রুৱ ভিন্ন * quant_weights,
        ধ্রুৱ পূর্ণাঙ্ক ntry) {

    ভিন্ন sigma2 = 0;
    বাবে (পূর্ণাঙ্ক j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(q4, 0, super_block_size/2);
    dh[0] = GGML_FP32_TO_FP16(0.f);

    ভিন্ন max_scale = 0, amax_scale = 0;
    বাবে (পূর্ণাঙ্ক ib = 0; ib < super_block_size/block_size; ++ib) {
        ধ্রুৱ ভিন্ন * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        যদি (quant_weights) {
            ধ্রুৱ ভিন্ন * qw = quant_weights + ib*block_size;
            বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } অন্যথা {
            বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        ভিন্ন amax = 0, max = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
            ভিন্ন ax = fabsf(xb[j]);
            যদি (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        যদি (amax < GROUP_MAX_EPS) {
            scales[ib] = 0;
            অনুৱর্ত;
        }
        ভিন্ন d = ntry > 0 ? -max/মানসমূহ[0] : max/মানসমূহ[0];
        ভিন্ন id = 1/d;
        ভিন্ন sumqx = 0, sumq2 = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
            ভিন্ন al = id*xb[j];
            পূর্ণাঙ্ক l = best_index_int8(16, মানসমূহ, al);
            Lb[j] = l;
            ভিন্ন q = মানসমূহ[l];
            ভিন্ন w = weight[j];
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        d = sumq2 > 0 ? sumqx/sumq2 : 0.f;
        ভিন্ন best = d*sumqx;
        বাবে (পূর্ণাঙ্ক itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + মানসমূহ[0])/max;
            sumqx = sumq2 = 0;
            বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
                ভিন্ন al = id*xb[j];
                পূর্ণাঙ্ক l = best_index_int8(16, মানসমূহ, al);
                ভিন্ন q = মানসমূহ[l];
                ভিন্ন w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            যদি (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d * sumqx;
            }
        }
        scales[ib] = d;
        ভিন্ন abs_d = fabsf(d);
        যদি (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    যদি (super_block_size/block_size > 1) {
        পূর্ণাঙ্ক nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*পরিমাণ(uint16_t));
        ভিন্ন d = -max_scale/32;
        dh[0] = GGML_FP32_TO_FP16(d);
        ভিন্ন id = d ? 1/d : 0.f;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < super_block_size/block_size; ++ib) {
            পূর্ণাঙ্ক l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            ভিন্ন dl = d * l;
            ভিন্ন idl = dl ? 1/dl : 0.f;
            uint8_t * Lb = L + ib*block_size;
            ধ্রুৱ ভিন্ন * xb = x + ib*block_size;
            বাবে (পূর্ণাঙ্ক j = 0; j < block_size; ++j) {
                Lb[j] = best_index_int8(16, মানসমূহ, idl*xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            যদি (ib%2 == 0) scales_l[ib/2] = l_l;
            অন্যথা scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } অন্যথা {
        dh[0] = GGML_FP32_TO_FP16(scales[0]);
        যদি (ntry > 0) {
            ভিন্ন id = scales[0] ? 1/scales[0] : 0;
            বাবে (পূর্ণাঙ্ক j = 0; j < super_block_size; ++j) {
                L[j] = best_index_int8(16, মানসমূহ, id*x[j]);
            }
        }
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < super_block_size/32; ++i) {
        বাবে (পূর্ণাঙ্ক j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

size_t quantize_iq4_nl(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK4_NL == 0);
    int64_t nblock = n_per_row/QK4_NL;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    uint8_t L[QK4_NL];
    ভিন্ন weight[QK4_NL];
    uint16_t unused_h;
    uint8_t * unused_l = রিক্ত;
    ভিন্ন scale;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        block_iq4_nl * iq4 = (block_iq4_nl *)qrow;
        বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nblock; ++ibl) {
            ধ্রুৱ ভিন্ন * qw = quant_weights ? quant_weights + QK4_NL*ibl : রিক্ত;
            quantize_row_iq4_nl_impl(QK4_NL, 32, src + QK4_NL*ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                    &scale, weight, L, kvalues_iq4nl, qw, 7);
        }
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq4_nl);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq4_nl);
}

//void quantize_row_iq4_nl_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
শূন্য quantize_row_iq4_nl_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq4_nl * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k%QK4_NL == 0);
    int64_t nblock = k/QK4_NL;
    uint8_t L[QK4_NL];
    ভিন্ন weight[QK4_NL];
    uint16_t unused_h;
    uint8_t * unused_l = রিক্ত;
    ভিন্ন scale;
    block_iq4_nl * iq4 = y;
    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nblock; ++ibl) {
        quantize_row_iq4_nl_impl(QK4_NL, 32, x + QK4_NL*ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                &scale, weight, L, kvalues_iq4nl, রিক্ত, -1);
    }
}

size_t quantize_iq4_xs(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    uint8_t L[QK_K];
    ভিন্ন weight[32];
    ভিন্ন scales[QK_K/32];
    বাবে (int64_t row = 0; row < nrow; ++row) {
        block_iq4_xs * iq4 = (block_iq4_xs *)qrow;
        বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nblock; ++ibl) {
            ধ্রুৱ ভিন্ন * qw = quant_weights ? quant_weights + QK_K*ibl : রিক্ত;
            quantize_row_iq4_nl_impl(QK_K, 32, src + QK_K*ibl, &iq4[ibl].d, iq4[ibl].qs, &iq4[ibl].scales_h, iq4[ibl].scales_l,
                    scales, weight, L, kvalues_iq4nl, qw, 7);
        }
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq4_xs);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq4_xs);
}

শূন্য quantize_row_iq4_xs_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq4_xs * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, রিক্ত);
}

// =============================== 2.5625 bpw

স্থির শূন্য quantize_row_iq2_s_impl(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t n, ধ্রুৱ ভিন্ন * GGML_RESTRICT quant_weights) {

    ধ্রুৱ পূর্ণাঙ্ক gindex = iq2_data_index(GGML_TYPE_IQ2_S);

    ধ্রুৱ uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    ধ্রুৱ পূর্ণাঙ্ক      * kmap_q2xs       = iq2_data[gindex].map;
    ধ্রুৱ uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    ধ্রুৱ পূর্ণাঙ্ক kMaxQ = 3;

    ধ্রুৱ int64_t nbl = n/QK_K;

    block_iq2_s * y = vy;

    ভিন্ন scales[QK_K/16];
    ভিন্ন weight[16];
    ভিন্ন xval[16];
    int8_t L[16];
    int8_t Laux[16];
    ভিন্ন  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];

    বাবে (পূর্ণাঙ্ক ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, পরিমাণ(block_iq2_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        ভিন্ন max_scale = 0;

        ধ্রুৱ ভিন্ন * xbl = x + QK_K*ibl;
        ভিন্ন sumx2 = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        ভিন্ন sigma2 = 2*sumx2/QK_K;

        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {
            ধ্রুৱ ভিন্ন * xb = xbl + 16*ib;
            যদি (quant_weights) {
                ধ্রুৱ ভিন্ন * qw = quant_weights + QK_K*ibl + 16*ib;
                বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } অন্যথা {
                বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) weight[i] = 0.25f*sigma2 + xb[i]*xb[i];
            }
            বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                uint8_t s = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                    যদি (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    অন্যথা {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            ভিন্ন max = xval[0];
            বাবে (পূর্ণাঙ্ক i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            যদি (max < GROUP_MAX_EPS_IQ2_S) {
                scales[ib] = 0;
                অনুৱর্ত;
            }
            ভিন্ন best = 0;
            ভিন্ন scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            বাবে (পূর্ণাঙ্ক is = -9; is <= 9; ++is) {
                ভিন্ন id = (2*kMaxQ-1+is*0.1f)/max;
                ভিন্ন this_scale = 1/id;
                বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    যদি (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) L[i] = Laux[i];
                    বাবে (পূর্ণাঙ্ক k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            পূর্ণাঙ্ক n_not_ongrid = 0;
            বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) যদি (!is_on_grid[k]) ++n_not_ongrid;
            যদি (n_not_ongrid > 0 && scale > 0) {
                ভিন্ন id = 1/scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                    যদি (is_on_grid[k]) অনুৱর্ত;
                    uint16_t u = 0;
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) {
                        পূর্ণাঙ্ক l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                    যদি (grid_index < 0) {
                        ধ্রুৱ uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                ভিন্ন sumqx = 0, sumq2 = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 16; ++i) {
                    ভিন্ন w = weight[i];
                    ভিন্ন q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                যদি (sumq2 > 0) scale = sumqx/sumq2;
            }
            যদি (scale < 0) {
                scale = -scale;
                বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) block_signs[k] = ~block_signs[k];
            }
            বাবে (পূর্ণাঙ্ক k = 0; k < 2; ++k) {
                uint16_t u = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                পূর্ণাঙ্ক grid_index = kmap_q2xs[u];
                যদি (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    বাবে (পূর্ণাঙ্ক i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                ধ্রুৱ পূর্ণাঙ্ক i8 = 2*ib + k;
                y[ibl].qs[i8] = grid_index & 255;
                y[ibl].qh[i8/4] |= ((grid_index >> 8) << 2*(i8%4));
                y[ibl].qs[QK_K/8 + i8] = block_signs[k];
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        যদি (!max_scale) {
            অনুৱর্ত;
        }

        ভিন্ন d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 0.9875f);
        ভিন্ন id = 1/d;
        বাবে (পূর্ণাঙ্ক ib = 0; ib < QK_K/16; ++ib) {
            পূর্ণাঙ্ক l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            যদি (ib%2 == 0) y[ibl].scales[ib/2] = l;
            অন্যথা y[ibl].scales[ib/2] |= (l << 4);
        }
    }
}

size_t quantize_iq2_s(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, ধ্রুৱ ভিন্ন * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    ৱর্ণ * qrow = (ৱর্ণ *)dst;
    বাবে (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_s_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*পরিমাণ(block_iq2_s);
    }
    ঘূৰাই nrow * nblock * পরিমাণ(block_iq2_s);
}

শূন্য quantize_row_iq2_s_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq2_s * GGML_RESTRICT y, int64_t k) {
    প্রতিজ্ঞা(k % QK_K == 0);
    quantize_iq2_s(x, y, 1, k, রিক্ত);
}

// =============================== data validation

স্থির bool validate_float(ভিন্ন f, size_t i) {
    যদি (isinf(f)) {
        fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i);
        ঘূৰাই false;
    }

    যদি (isnan(f)) {
        fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i);
        ঘূৰাই false;
    }

    ঘূৰাই true;
}

স্থির bool isinf_fp16(ggml_fp16_t f) {
    ঘূৰাই (f & 0x7c00) == 0x7c00 && (f & 0x03ff) == 0;
}

স্থির bool isnan_fp16(ggml_fp16_t f) {
    ঘূৰাই (f & 0x7c00) == 0x7c00 && (f & 0x03ff) != 0;
}

স্থির bool validate_fp16(ggml_fp16_t f, size_t i) {
    যদি (isinf_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i);
        ঘূৰাই false;
    }

    যদি (isnan_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i);
        ঘূৰাই false;
    }

    ঘূৰাই true;
}

স্থির bool validate_e_e8m0(uint8_t e, size_t i) {
    যদি (e == 0xff) {
        fprintf(stderr, "ggml_validate_row_data: found invalid e value %d at block %zu\n", e, i);
        ঘূৰাই false;
    }

    ঘূৰাই true;
}

#define VALIDATE_ROW_DATA_D_F16_IMPL(type, data, nb) \
    ধ্রুৱ type * q = (ধ্রুৱ type *) (data); \
    বাবে (size_t i = 0; i < (nb); ++i) { \
        যদি (!validate_fp16(q[i].d, i)) { \
            ঘূৰাই false; \
        } \
    }

#define VALIDATE_ROW_DATA_DM_F16_IMPL(type, data, nb, d, m) \
    ধ্রুৱ type * q = (ধ্রুৱ type *) (data); \
    বাবে (size_t i = 0; i < (nb); ++i) { \
        যদি (!validate_fp16(q[i].d, i) || !validate_fp16(q[i].m, i)) { \
            ঘূৰাই false; \
        } \
    }

#define VALIDATE_ROW_DATA_E_E8M0_IMPL(type, data, nb) \
    ধ্রুৱ type * q = (ধ্রুৱ type *) (data); \
    বাবে (size_t i = 0; i < (nb); ++i) { \
        যদি (!validate_e_e8m0(q[i].e, i)) { \
            ঘূৰাই false; \
        } \
    }

#define VALIDATE_ROW_DATA_DVEC_F16_IMPL(type, data, nb, nr) \
    ধ্রুৱ type * q = (ধ্রুৱ type *) (data); \
    বাবে (size_t i = 0; i < (nb); ++i) { \
        বাবে (size_t j = 0; j < (nr); ++j) { \
            যদি (!validate_fp16(q[i].d[j], i)) { \
                ঘূৰাই false; \
            } \
        } \
    }

bool ggml_validate_row_data(গণনা ggml_type type, ধ্রুৱ শূন্য * data, size_t nbytes) {
    যদি (type < 0 || type >= GGML_TYPE_গুনতি) {
        fprintf(stderr, "%s: invalid type %d\n", __func__, type);
        ঘূৰাই false;
    }

    যদি (nbytes % ggml_type_size(type) != 0) {
        fprintf(stderr, "%s: invalid size %zu for type %s (type size = %zu)\n", __func__, nbytes, ggml_type_name(type), ggml_type_size(type));
        ঘূৰাই false;
    }

    ধ্রুৱ size_t nb = nbytes/ggml_type_size(type);

    ৱিকল্প (type) {
        পক্ষ GGML_TYPE_BF16:
            {
                পূর্ণাঙ্ক nans = 0;
                পূর্ণাঙ্ক infs = 0;
                ধ্রুৱ unsigned short * f = (ধ্রুৱ unsigned short *) data;
                বাবে (size_t i = 0; i < nb; ++i) {
                    nans += (f[i] & 0x7fff) > 0x7f80;
                    infs += (f[i] & 0x7fff) == 0x7f80;
                }
                যদি (nans) {
                    fprintf(stderr, "%s: found %d NaNs in row of %zu BF16 values\n", __func__, nans, nb);
                    ঘূৰাই false;
                }
                যদি (infs) {
                    fprintf(stderr, "%s: found %d infinities in row of %zu BF16 values\n", __func__, infs, nb);
                    ঘূৰাই false;
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ধ্রুৱ ggml_fp16_t * f = (ধ্রুৱ ggml_fp16_t *) data;
                size_t i = 0;
#যদি defined(__AVX2__)
                বাবে (; i + 15 < nb; i += 16) {
                    __m256i v = _mm256_loadu_si256((ধ্রুৱ __m256i *)(f + i));
                    __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi16(0x7c00));
                    __m256i cmp = _mm256_cmpeq_epi16(vexp, _mm256_set1_epi16(0x7c00));
                    পূর্ণাঙ্ক mask = _mm256_movemask_epi8(cmp);
                    যদি (mask) {
                        বাবে (size_t j = 0; j < 16; ++j) {
                            যদি (!validate_fp16(f[i + j], i + j)) {
                                ঘূৰাই false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#নোচেদ্যদি defined(__ARM_NEON)
                বাবে (; i + 7 < nb; i += 8) {
                    uint16x8_t v = vld1q_u16(f + i);
                    uint16x8_t vexp = vandq_u16(v, vdupq_n_u16(0x7c00));
                    uint16x8_t cmp = vceqq_u16(vexp, vdupq_n_u16(0x7c00));
                    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(cmp, 4)), 0);
                    যদি (mask) {
                        বাবে (size_t j = 0; j < 8; ++j) {
                            যদি (!validate_fp16(f[i + j], i + j)) {
                                ঘূৰাই false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#endif
                বাবে (; i < nb; ++i) {
                    যদি (!validate_fp16(f[i], i)) {
                        ঘূৰাই false;
                    }
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ধ্রুৱ ভিন্ন * f = (ধ্রুৱ ভিন্ন *) data;
                size_t i = 0;
#যদি defined(__AVX2__)
                বাবে (; i + 7 < nb; i += 8) {
                    __m256i v = _mm256_loadu_si256((ধ্রুৱ __m256i *)(f + i));
                    __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi32(0x7f800000));
                    __m256i cmp = _mm256_cmpeq_epi32(vexp, _mm256_set1_epi32(0x7f800000));
                    পূর্ণাঙ্ক mask = _mm256_movemask_epi8(cmp);
                    যদি (mask) {
                        বাবে (size_t j = 0; j < 8; ++j) {
                            যদি (!validate_float(f[i + j], i + j)) {
                                ঘূৰাই false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#নোচেদ্যদি defined(__ARM_NEON)
                বাবে (; i + 3 < nb; i += 4) {
                    uint32x4_t v = vld1q_u32((ধ্রুৱ uint32_t *)f + i);
                    uint32x4_t vexp = vandq_u32(v, vdupq_n_u32(0x7f800000));
                    uint32x4_t cmp = vceqq_u32(vexp, vdupq_n_u32(0x7f800000));
                    uint64_t mask = vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(cmp, 8)), 0);
                    যদি (mask) {
                        বাবে (size_t j = 0; j < 4; ++j) {
                            যদি (!validate_float(f[i + j], i + j)) {
                                ঘূৰাই false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#endif
                বাবে (; i < nb; ++i) {
                    যদি (!validate_float(f[i], i)) {
                        ঘূৰাই false;
                    }
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F64:
            {
                ধ্রুৱ দ্ৱিগুণ * f = (ধ্রুৱ দ্ৱিগুণ *) data;
                বাবে (size_t i = 0; i < nb; ++i) {
                    যদি (!validate_float(f[i], i)) {
                        ঘূৰাই false;
                    }
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q1_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q1_0, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q2_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q2_0, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q4_0, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_1:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_1, data, nb, d, m);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q5_0, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_1:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_1, data, nb, d, m);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q8_0, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_MXFP4:
            {
                VALIDATE_ROW_DATA_E_E8M0_IMPL(block_mxfp4, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_NVFP4:
            {
                // UE4M3 scales are uint8_t — all byte values are valid
                GGML_UNUSED(data);
                GGML_UNUSED(nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q2_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q2_K, data, nb, d, dmin);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q3_K:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q3_K, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_K, data, nb, d, dmin);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_K, data, nb, d, dmin);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q6_K:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q6_K, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_K:
            {
                ধ্রুৱ block_q8_K * q = (ধ্রুৱ block_q8_K *) data;
                বাবে (size_t i = 0; i < nb; ++i) {
                    যদি (!validate_float(q[i].d, i)) {
                        ঘূৰাই false;
                    }
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_TQ1_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq1_0, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_TQ2_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_0, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ1_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq1_s, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ1_M:
            {
                ধ্রুৱ block_iq1_m * q = (ধ্রুৱ block_iq1_m *) data;
                বাবে (size_t i = 0; i < nb; ++i) {
                    iq1m_scale_t scale;
                    ধ্রুৱ uint16_t * sc = (ধ্রুৱ uint16_t *)q[i].scales;
                    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                    যদি (!validate_fp16(scale.f16, i)) {
                        ঘূৰাই false;
                    }
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_XXS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xxs, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_XS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xs, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_s, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ3_XXS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_xxs, data, nb);
            } ভঙ্গ;

        পক্ষ GGML_TYPE_IQ3_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_s, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ4_XS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_xs, data, nb);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_IQ4_NL:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_nl, data, nb);
            } ভঙ্গ;

        পক্ষ GGML_TYPE_I8:
        পক্ষ GGML_TYPE_I16:
        পক্ষ GGML_TYPE_I32:
        পক্ষ GGML_TYPE_I64:
            // nothing to validate
            ভঙ্গ;
        স্ৱভাৱ:
            {
                fprintf(stderr, "%s: invalid type %d\n", __func__, type);
                ঘূৰাই false;
            }
    }

    ঘূৰাই true;
}
