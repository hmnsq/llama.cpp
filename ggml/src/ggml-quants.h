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
GGML_API शून्य quantize_row_q1_0_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q2_0_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_0_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_1_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_0_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_1_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_0_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_1_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_mxfp4_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_nvfp4_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_q2_K_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q3_K_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_K_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_K_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q6_K_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_K_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_tq1_0_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_tq2_0_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_iq3_xxs_ref(स्थिरांक दशांश * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq4_nl_ref (स्थिरांक दशांश * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq4_xs_ref (स्थिरांक दशांश * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq3_s_ref  (स्थिरांक दशांश * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq2_s_ref  (स्थिरांक दशांश * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API शून्य dequantize_row_q1_0(स्थिरांक block_q1_0 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q2_0(स्थिरांक block_q2_0 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_0(स्थिरांक block_q4_0 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_1(स्थिरांक block_q4_1 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_0(स्थिरांक block_q5_0 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_1(स्थिरांक block_q5_1 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q8_0(स्थिरांक block_q8_0 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_mxfp4(स्थिरांक block_mxfp4 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_nvfp4(स्थिरांक block_nvfp4 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_q2_K(स्थिरांक block_q2_K * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q3_K(स्थिरांक block_q3_K * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_K(स्थिरांक block_q4_K * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_K(स्थिरांक block_q5_K * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q6_K(स्थिरांक block_q6_K * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q8_K(स्थिरांक block_q8_K * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_tq1_0(स्थिरांक block_tq1_0 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_tq2_0(स्थिरांक block_tq2_0 * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_iq2_xxs(स्थिरांक block_iq2_xxs * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq2_xs (स्थिरांक block_iq2_xs  * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq2_s  (स्थिरांक block_iq2_s   * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq3_xxs(स्थिरांक block_iq3_xxs * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq1_s  (स्थिरांक block_iq1_s   * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq1_m  (स्थिरांक block_iq1_m   * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq4_nl (स्थिरांक block_iq4_nl  * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq4_xs (स्थिरांक block_iq4_xs  * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq3_s  (स्थिरांक block_iq3_s   * GGML_RESTRICT x, दशांश * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API माप_प्रकार quantize_iq2_xxs(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq2_xs (स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq2_s  (स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq3_xxs(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq1_s  (स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq1_m  (स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq4_nl (स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq4_xs (स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_iq3_s  (स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);

GGML_API माप_प्रकार quantize_tq1_0(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_tq2_0(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);

GGML_API माप_प्रकार quantize_q2_K(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q3_K(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q4_K(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q5_K(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q6_K(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q1_0(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q2_0(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q4_0(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q4_1(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q5_0(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q5_1(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_q8_0(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);

GGML_API माप_प्रकार quantize_mxfp4(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);
GGML_API माप_प्रकार quantize_nvfp4(स्थिरांक दशांश * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, स्थिरांक दशांश * imatrix);

GGML_API शून्य iq2xs_init_impl(गणना ggml_type type);
GGML_API शून्य iq2xs_free_impl(गणना ggml_type type);
GGML_API शून्य iq3xs_init_impl(पूर्णांक grid_size);
GGML_API शून्य iq3xs_free_impl(पूर्णांक grid_size);

#ifdef __cplusplus
}
#endif
