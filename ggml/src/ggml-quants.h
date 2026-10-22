#pragma once

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"

#ضمن "ggml.h"

// GGML internal header

#ifdef __cplusplus
خارجي "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API فراغ quantize_row_q1_0_ref(ثابت عائم * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q2_0_ref(ثابت عائم * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q4_0_ref(ثابت عائم * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q4_1_ref(ثابت عائم * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q5_0_ref(ثابت عائم * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q5_1_ref(ثابت عائم * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q8_0_ref(ثابت عائم * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q8_1_ref(ثابت عائم * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_row_mxfp4_ref(ثابت عائم * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_nvfp4_ref(ثابت عائم * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_row_q2_K_ref(ثابت عائم * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q3_K_ref(ثابت عائم * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q4_K_ref(ثابت عائم * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q5_K_ref(ثابت عائم * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q6_K_ref(ثابت عائم * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_q8_K_ref(ثابت عائم * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_row_tq1_0_ref(ثابت عائم * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_tq2_0_ref(ثابت عائم * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_row_iq3_xxs_ref(ثابت عائم * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_iq4_nl_ref (ثابت عائم * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_iq4_xs_ref (ثابت عائم * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_iq3_s_ref  (ثابت عائم * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_row_iq2_s_ref  (ثابت عائم * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API فراغ dequantize_row_q1_0(ثابت block_q1_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q2_0(ثابت block_q2_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q4_0(ثابت block_q4_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q4_1(ثابت block_q4_1 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q5_0(ثابت block_q5_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q5_1(ثابت block_q5_1 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q8_0(ثابت block_q8_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_row_mxfp4(ثابت block_mxfp4 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_nvfp4(ثابت block_nvfp4 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_row_q2_K(ثابت block_q2_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q3_K(ثابت block_q3_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q4_K(ثابت block_q4_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q5_K(ثابت block_q5_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q6_K(ثابت block_q6_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_q8_K(ثابت block_q8_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_row_tq1_0(ثابت block_tq1_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_tq2_0(ثابت block_tq2_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_row_iq2_xxs(ثابت block_iq2_xxs * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq2_xs (ثابت block_iq2_xs  * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq2_s  (ثابت block_iq2_s   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq3_xxs(ثابت block_iq3_xxs * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq1_s  (ثابت block_iq1_s   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq1_m  (ثابت block_iq1_m   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq4_nl (ثابت block_iq4_nl  * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq4_xs (ثابت block_iq4_xs  * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_row_iq3_s  (ثابت block_iq3_s   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq2_xs (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq2_s  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq3_xxs(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq1_s  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq1_m  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq4_nl (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq4_xs (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_iq3_s  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);

GGML_API size_t quantize_tq1_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_tq2_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);

GGML_API size_t quantize_q2_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q3_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q4_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q5_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q6_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q1_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q2_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q4_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q4_1(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q5_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q5_1(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_q8_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);

GGML_API size_t quantize_mxfp4(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);
GGML_API size_t quantize_nvfp4(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ثابت عائم * imatrix);

GGML_API فراغ iq2xs_init_impl(تعداد ggml_type type);
GGML_API فراغ iq2xs_free_impl(تعداد ggml_type type);
GGML_API فراغ iq3xs_init_impl(صحيح grid_size);
GGML_API فراغ iq3xs_free_impl(صحيح grid_size);

#ifdef __cplusplus
}
#endif
