#pragma once

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "ggml.h"

// GGML internal header

#ifdef __cplusplus
बाह्य "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API शून्य quantize_row_q1_0_ref(अचर भिन्न * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q2_0_ref(अचर भिन्न * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_0_ref(अचर भिन्न * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_1_ref(अचर भिन्न * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_0_ref(अचर भिन्न * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_1_ref(अचर भिन्न * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_0_ref(अचर भिन्न * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_1_ref(अचर भिन्न * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_mxfp4_ref(अचर भिन्न * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_nvfp4_ref(अचर भिन्न * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_q2_K_ref(अचर भिन्न * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q3_K_ref(अचर भिन्न * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_K_ref(अचर भिन्न * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_K_ref(अचर भिन्न * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q6_K_ref(अचर भिन्न * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_K_ref(अचर भिन्न * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_tq1_0_ref(अचर भिन्न * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_tq2_0_ref(अचर भिन्न * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_iq3_xxs_ref(अचर भिन्न * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq4_nl_ref (अचर भिन्न * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq4_xs_ref (अचर भिन्न * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq3_s_ref  (अचर भिन्न * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq2_s_ref  (अचर भिन्न * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API शून्य dequantize_row_q1_0(अचर block_q1_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q2_0(अचर block_q2_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_0(अचर block_q4_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_1(अचर block_q4_1 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_0(अचर block_q5_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_1(अचर block_q5_1 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q8_0(अचर block_q8_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_mxfp4(अचर block_mxfp4 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_nvfp4(अचर block_nvfp4 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_q2_K(अचर block_q2_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q3_K(अचर block_q3_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_K(अचर block_q4_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_K(अचर block_q5_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q6_K(अचर block_q6_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q8_K(अचर block_q8_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_tq1_0(अचर block_tq1_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_tq2_0(अचर block_tq2_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_iq2_xxs(अचर block_iq2_xxs * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq2_xs (अचर block_iq2_xs  * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq2_s  (अचर block_iq2_s   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq3_xxs(अचर block_iq3_xxs * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq1_s  (अचर block_iq1_s   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq1_m  (अचर block_iq1_m   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq4_nl (अचर block_iq4_nl  * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq4_xs (अचर block_iq4_xs  * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq3_s  (अचर block_iq3_s   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API माप_प्रकार quantize_iq2_xxs(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq2_xs (अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq2_s  (अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq3_xxs(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq1_s  (अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq1_m  (अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq4_nl (अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq4_xs (अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_iq3_s  (अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);

GGML_API माप_प्रकार quantize_tq1_0(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_tq2_0(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);

GGML_API माप_प्रकार quantize_q2_K(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q3_K(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q4_K(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q5_K(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q6_K(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q1_0(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q2_0(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q4_0(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q4_1(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q5_0(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q5_1(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_q8_0(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);

GGML_API माप_प्रकार quantize_mxfp4(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);
GGML_API माप_प्रकार quantize_nvfp4(अचर भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, अचर भिन्न * imatrix);

GGML_API शून्य iq2xs_init_impl(गणना ggml_type type);
GGML_API शून्य iq2xs_free_impl(गणना ggml_type type);
GGML_API शून्य iq3xs_init_impl(पूर्णाङ्क grid_size);
GGML_API शून्य iq3xs_free_impl(पूर्णाङ्क grid_size);

#ifdef __cplusplus
}
#endif
