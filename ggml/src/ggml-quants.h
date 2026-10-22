#pragma once

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"

#ଅନ୍ତର୍ଭାଵ "ggml.h"

// GGML internal header

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API ଶୂନ୍ୟ quantize_row_q1_0_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q2_0_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q4_0_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q4_1_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q5_0_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q5_1_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q8_0_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q8_1_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ quantize_row_mxfp4_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_nvfp4_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ quantize_row_q2_K_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q3_K_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q4_K_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q5_K_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q6_K_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_q8_K_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ quantize_row_tq1_0_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_tq2_0_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ quantize_row_iq3_xxs_ref(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_iq4_nl_ref (ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_iq4_xs_ref (ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_iq3_s_ref  (ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ quantize_row_iq2_s_ref  (ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API ଶୂନ୍ୟ dequantize_row_q1_0(ଅଚଳ block_q1_0 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q2_0(ଅଚଳ block_q2_0 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q4_0(ଅଚଳ block_q4_0 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q4_1(ଅଚଳ block_q4_1 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q5_0(ଅଚଳ block_q5_0 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q5_1(ଅଚଳ block_q5_1 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q8_0(ଅଚଳ block_q8_0 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ dequantize_row_mxfp4(ଅଚଳ block_mxfp4 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_nvfp4(ଅଚଳ block_nvfp4 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ dequantize_row_q2_K(ଅଚଳ block_q2_K * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q3_K(ଅଚଳ block_q3_K * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q4_K(ଅଚଳ block_q4_K * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q5_K(ଅଚଳ block_q5_K * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q6_K(ଅଚଳ block_q6_K * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_q8_K(ଅଚଳ block_q8_K * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ dequantize_row_tq1_0(ଅଚଳ block_tq1_0 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_tq2_0(ଅଚଳ block_tq2_0 * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);

GGML_API ଶୂନ୍ୟ dequantize_row_iq2_xxs(ଅଚଳ block_iq2_xxs * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq2_xs (ଅଚଳ block_iq2_xs  * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq2_s  (ଅଚଳ block_iq2_s   * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq3_xxs(ଅଚଳ block_iq3_xxs * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq1_s  (ଅଚଳ block_iq1_s   * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq1_m  (ଅଚଳ block_iq1_m   * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq4_nl (ଅଚଳ block_iq4_nl  * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq4_xs (ଅଚଳ block_iq4_xs  * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);
GGML_API ଶୂନ୍ୟ dequantize_row_iq3_s  (ଅଚଳ block_iq3_s   * GGML_RESTRICT x, ଦଶମିକ * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq2_xs (ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq2_s  (ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq3_xxs(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq1_s  (ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq1_m  (ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq4_nl (ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq4_xs (ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_iq3_s  (ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);

GGML_API size_t quantize_tq1_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_tq2_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);

GGML_API size_t quantize_q2_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q3_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q4_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q5_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q6_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q1_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q2_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q4_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q4_1(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q5_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q5_1(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_q8_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);

GGML_API size_t quantize_mxfp4(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);
GGML_API size_t quantize_nvfp4(ଅଚଳ ଦଶମିକ * GGML_RESTRICT src, ଶୂନ୍ୟ * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ଅଚଳ ଦଶମିକ * imatrix);

GGML_API ଶୂନ୍ୟ iq2xs_init_impl(ଗଣନା ggml_type type);
GGML_API ଶୂନ୍ୟ iq2xs_free_impl(ଗଣନା ggml_type type);
GGML_API ଶୂନ୍ୟ iq3xs_init_impl(ପୂର୍ଣ୍ଣସଂଖ୍ୟା grid_size);
GGML_API ଶୂନ୍ୟ iq3xs_free_impl(ପୂର୍ଣ୍ଣସଂଖ୍ୟା grid_size);

#ifdef __cplusplus
}
#endif
