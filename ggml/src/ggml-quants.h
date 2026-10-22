#pragma once

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"

#அந்தர்பாவ "ggml.h"

// GGML internal header

#ifdef __cplusplus
வெளி "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API வெறுமை quantize_row_q1_0_ref(மாறிலி தசம * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q2_0_ref(மாறிலி தசம * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q4_0_ref(மாறிலி தசம * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q4_1_ref(மாறிலி தசம * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q5_0_ref(மாறிலி தசம * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q5_1_ref(மாறிலி தசம * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q8_0_ref(மாறிலி தசம * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q8_1_ref(மாறிலி தசம * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை quantize_row_mxfp4_ref(மாறிலி தசம * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_nvfp4_ref(மாறிலி தசம * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை quantize_row_q2_K_ref(மாறிலி தசம * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q3_K_ref(மாறிலி தசம * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q4_K_ref(மாறிலி தசம * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q5_K_ref(மாறிலி தசம * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q6_K_ref(மாறிலி தசம * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_q8_K_ref(மாறிலி தசம * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை quantize_row_tq1_0_ref(மாறிலி தசம * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_tq2_0_ref(மாறிலி தசம * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை quantize_row_iq3_xxs_ref(மாறிலி தசம * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_iq4_nl_ref (மாறிலி தசம * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_iq4_xs_ref (மாறிலி தசம * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_iq3_s_ref  (மாறிலி தசம * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை quantize_row_iq2_s_ref  (மாறிலி தசம * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API வெறுமை dequantize_row_q1_0(மாறிலி block_q1_0 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q2_0(மாறிலி block_q2_0 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q4_0(மாறிலி block_q4_0 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q4_1(மாறிலி block_q4_1 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q5_0(மாறிலி block_q5_0 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q5_1(மாறிலி block_q5_1 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q8_0(மாறிலி block_q8_0 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை dequantize_row_mxfp4(மாறிலி block_mxfp4 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_nvfp4(மாறிலி block_nvfp4 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை dequantize_row_q2_K(மாறிலி block_q2_K * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q3_K(மாறிலி block_q3_K * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q4_K(மாறிலி block_q4_K * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q5_K(மாறிலி block_q5_K * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q6_K(மாறிலி block_q6_K * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_q8_K(மாறிலி block_q8_K * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை dequantize_row_tq1_0(மாறிலி block_tq1_0 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_tq2_0(மாறிலி block_tq2_0 * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);

GGML_API வெறுமை dequantize_row_iq2_xxs(மாறிலி block_iq2_xxs * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq2_xs (மாறிலி block_iq2_xs  * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq2_s  (மாறிலி block_iq2_s   * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq3_xxs(மாறிலி block_iq3_xxs * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq1_s  (மாறிலி block_iq1_s   * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq1_m  (மாறிலி block_iq1_m   * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq4_nl (மாறிலி block_iq4_nl  * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq4_xs (மாறிலி block_iq4_xs  * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);
GGML_API வெறுமை dequantize_row_iq3_s  (மாறிலி block_iq3_s   * GGML_RESTRICT x, தசம * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq2_xs (மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq2_s  (மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq3_xxs(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq1_s  (மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq1_m  (மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq4_nl (மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq4_xs (மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_iq3_s  (மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);

GGML_API size_t quantize_tq1_0(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_tq2_0(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);

GGML_API size_t quantize_q2_K(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q3_K(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q4_K(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q5_K(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q6_K(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q1_0(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q2_0(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q4_0(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q4_1(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q5_0(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q5_1(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_q8_0(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);

GGML_API size_t quantize_mxfp4(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);
GGML_API size_t quantize_nvfp4(மாறிலி தசம * GGML_RESTRICT src, வெறுமை * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, மாறிலி தசம * imatrix);

GGML_API வெறுமை iq2xs_init_impl(எண்ணிக்கை ggml_type type);
GGML_API வெறுமை iq2xs_free_impl(எண்ணிக்கை ggml_type type);
GGML_API வெறுமை iq3xs_init_impl(முழு grid_size);
GGML_API வெறுமை iq3xs_free_impl(முழு grid_size);

#ifdef __cplusplus
}
#endif
