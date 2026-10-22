#pragma once

#define GGML_COMMON_DECL_C
#অন্তর্ভাৱ "ggml-common.h"

#অন্তর্ভাৱ "ggml.h"

// GGML internal header

#ifdef __cplusplus
বাহ্য "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API শূন্য quantize_row_q1_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q2_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q4_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q4_1_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q5_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q5_1_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q8_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q8_1_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য quantize_row_mxfp4_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_nvfp4_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য quantize_row_q2_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q3_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q4_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q5_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q6_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_q8_K_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য quantize_row_tq1_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_tq2_0_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য quantize_row_iq3_xxs_ref(ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_iq4_nl_ref (ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_iq4_xs_ref (ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_iq3_s_ref  (ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য quantize_row_iq2_s_ref  (ধ্রুৱ ভিন্ন * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API শূন্য dequantize_row_q1_0(ধ্রুৱ block_q1_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q2_0(ধ্রুৱ block_q2_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q4_0(ধ্রুৱ block_q4_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q4_1(ধ্রুৱ block_q4_1 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q5_0(ধ্রুৱ block_q5_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q5_1(ধ্রুৱ block_q5_1 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q8_0(ধ্রুৱ block_q8_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য dequantize_row_mxfp4(ধ্রুৱ block_mxfp4 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_nvfp4(ধ্রুৱ block_nvfp4 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য dequantize_row_q2_K(ধ্রুৱ block_q2_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q3_K(ধ্রুৱ block_q3_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q4_K(ধ্রুৱ block_q4_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q5_K(ধ্রুৱ block_q5_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q6_K(ধ্রুৱ block_q6_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_q8_K(ধ্রুৱ block_q8_K * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য dequantize_row_tq1_0(ধ্রুৱ block_tq1_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_tq2_0(ধ্রুৱ block_tq2_0 * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);

GGML_API শূন্য dequantize_row_iq2_xxs(ধ্রুৱ block_iq2_xxs * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq2_xs (ধ্রুৱ block_iq2_xs  * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq2_s  (ধ্রুৱ block_iq2_s   * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq3_xxs(ধ্রুৱ block_iq3_xxs * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq1_s  (ধ্রুৱ block_iq1_s   * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq1_m  (ধ্রুৱ block_iq1_m   * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq4_nl (ধ্রুৱ block_iq4_nl  * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq4_xs (ধ্রুৱ block_iq4_xs  * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);
GGML_API শূন্য dequantize_row_iq3_s  (ধ্রুৱ block_iq3_s   * GGML_RESTRICT x, ভিন্ন * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq2_xs (ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq2_s  (ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq3_xxs(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq1_s  (ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq1_m  (ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq4_nl (ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq4_xs (ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_iq3_s  (ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);

GGML_API size_t quantize_tq1_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_tq2_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);

GGML_API size_t quantize_q2_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q3_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q4_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q5_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q6_K(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q1_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q2_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q4_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q4_1(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q5_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q5_1(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_q8_0(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);

GGML_API size_t quantize_mxfp4(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);
GGML_API size_t quantize_nvfp4(ধ্রুৱ ভিন্ন * GGML_RESTRICT src, শূন্য * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, ধ্রুৱ ভিন্ন * imatrix);

GGML_API শূন্য iq2xs_init_impl(গণনা ggml_type type);
GGML_API শূন্য iq2xs_free_impl(গণনা ggml_type type);
GGML_API শূন্য iq3xs_init_impl(পূর্ণাঙ্ক grid_size);
GGML_API শূন্য iq3xs_free_impl(পূর্ণাঙ্ক grid_size);

#ifdef __cplusplus
}
#endif
