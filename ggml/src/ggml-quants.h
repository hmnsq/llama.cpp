#pragma once

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"

// GGML internal header

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API ಶೂನ್ಯ quantize_row_q1_0_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q2_0_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q4_0_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q4_1_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q5_0_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q5_1_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q8_0_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q8_1_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_row_mxfp4_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_nvfp4_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_row_q2_K_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q3_K_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q4_K_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q5_K_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q6_K_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_q8_K_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_row_tq1_0_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_tq2_0_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_row_iq3_xxs_ref(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_iq4_nl_ref (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_iq4_xs_ref (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_iq3_s_ref  (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_row_iq2_s_ref  (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API ಶೂನ್ಯ dequantize_row_q1_0(ಧ್ರುವ block_q1_0 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q2_0(ಧ್ರುವ block_q2_0 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q4_0(ಧ್ರುವ block_q4_0 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q4_1(ಧ್ರುವ block_q4_1 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q5_0(ಧ್ರುವ block_q5_0 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q5_1(ಧ್ರುವ block_q5_1 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q8_0(ಧ್ರುವ block_q8_0 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_row_mxfp4(ಧ್ರುವ block_mxfp4 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_nvfp4(ಧ್ರುವ block_nvfp4 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_row_q2_K(ಧ್ರುವ block_q2_K * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q3_K(ಧ್ರುವ block_q3_K * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q4_K(ಧ್ರುವ block_q4_K * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q5_K(ಧ್ರುವ block_q5_K * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q6_K(ಧ್ರುವ block_q6_K * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_q8_K(ಧ್ರುವ block_q8_K * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_row_tq1_0(ಧ್ರುವ block_tq1_0 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_tq2_0(ಧ್ರುವ block_tq2_0 * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_row_iq2_xxs(ಧ್ರುವ block_iq2_xxs * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq2_xs (ಧ್ರುವ block_iq2_xs  * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq2_s  (ಧ್ರುವ block_iq2_s   * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq3_xxs(ಧ್ರುವ block_iq3_xxs * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq1_s  (ಧ್ರುವ block_iq1_s   * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq1_m  (ಧ್ರುವ block_iq1_m   * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq4_nl (ಧ್ರುವ block_iq4_nl  * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq4_xs (ಧ್ರುವ block_iq4_xs  * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_row_iq3_s  (ಧ್ರುವ block_iq3_s   * GGML_RESTRICT x, ಭಿನ್ನ * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq2_xs (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq2_s  (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq3_xxs(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq1_s  (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq1_m  (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq4_nl (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq4_xs (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_iq3_s  (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);

GGML_API size_t quantize_tq1_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_tq2_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);

GGML_API size_t quantize_q2_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q3_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q4_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q5_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q6_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q1_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q2_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q4_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q4_1(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q5_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q5_1(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_q8_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);

GGML_API size_t quantize_mxfp4(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);
GGML_API size_t quantize_nvfp4(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ಧ್ರುವ ಭಿನ್ನ * imatrix);

GGML_API ಶೂನ್ಯ iq2xs_init_impl(ಗಣನಾ ggml_type type);
GGML_API ಶೂನ್ಯ iq2xs_free_impl(ಗಣನಾ ggml_type type);
GGML_API ಶೂನ್ಯ iq3xs_init_impl(ಪೂರ್ಣಾಙ್ಕ grid_size);
GGML_API ಶೂನ್ಯ iq3xs_free_impl(ಪೂರ್ಣಾಙ್ಕ grid_size);

#ifdef __cplusplus
}
#endif
