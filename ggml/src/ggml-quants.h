#निर्देशः once

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "ggml.h"

// GGML internal header

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API शून्य quantize_row_q1_0_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q2_0_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_0_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_1_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_0_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_1_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_0_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_1_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_mxfp4_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_nvfp4_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_q2_K_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q3_K_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q4_K_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q5_K_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q6_K_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_q8_K_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_tq1_0_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_tq2_0_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API शून्य quantize_row_iq3_xxs_ref(ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq4_nl_ref (ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq4_xs_ref (ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq3_s_ref  (ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API शून्य quantize_row_iq2_s_ref  (ध्रुव भिन्न * GGML_RESTRICT x, खण्डम्_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API शून्य dequantize_row_q1_0(ध्रुव खण्डम्_q1_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q2_0(ध्रुव खण्डम्_q2_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_0(ध्रुव खण्डम्_q4_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_1(ध्रुव खण्डम्_q4_1 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_0(ध्रुव खण्डम्_q5_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_1(ध्रुव खण्डम्_q5_1 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q8_0(ध्रुव खण्डम्_q8_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_mxfp4(ध्रुव खण्डम्_mxfp4 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_nvfp4(ध्रुव खण्डम्_nvfp4 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_q2_K(ध्रुव खण्डम्_q2_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q3_K(ध्रुव खण्डम्_q3_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q4_K(ध्रुव खण्डम्_q4_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q5_K(ध्रुव खण्डम्_q5_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q6_K(ध्रुव खण्डम्_q6_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_q8_K(ध्रुव खण्डम्_q8_K * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_tq1_0(ध्रुव खण्डम्_tq1_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_tq2_0(ध्रुव खण्डम्_tq2_0 * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

GGML_API शून्य dequantize_row_iq2_xxs(ध्रुव खण्डम्_iq2_xxs * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq2_xs (ध्रुव खण्डम्_iq2_xs  * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq2_s  (ध्रुव खण्डम्_iq2_s   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq3_xxs(ध्रुव खण्डम्_iq3_xxs * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq1_s  (ध्रुव खण्डम्_iq1_s   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq1_m  (ध्रुव खण्डम्_iq1_m   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq4_nl (ध्रुव खण्डम्_iq4_nl  * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq4_xs (ध्रुव खण्डम्_iq4_xs  * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);
GGML_API शून्य dequantize_row_iq3_s  (ध्रुव खण्डम्_iq3_s   * GGML_RESTRICT x, भिन्न * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq2_xs (ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq2_s  (ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq3_xxs(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq1_s  (ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq1_m  (ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq4_nl (ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq4_xs (ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_iq3_s  (ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);

GGML_API size_t quantize_tq1_0(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_tq2_0(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);

GGML_API size_t quantize_q2_K(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q3_K(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q4_K(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q5_K(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q6_K(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q1_0(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q2_0(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q4_0(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q4_1(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q5_0(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q5_1(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_q8_0(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);

GGML_API size_t quantize_mxfp4(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);
GGML_API size_t quantize_nvfp4(ध्रुव भिन्न * GGML_RESTRICT src, शून्य * GGML_RESTRICT dst, int64_t nrows, int64_t n_प्रत्येकम्_row, ध्रुव भिन्न * imatrix);

GGML_API शून्य iq2xs_init_impl(गणना ggml_type type);
GGML_API शून्य iq2xs_free_impl(गणना ggml_type type);
GGML_API शून्य iq3xs_init_impl(पूर्णाङ्क grid_आकार);
GGML_API शून्य iq3xs_free_impl(पूर्णाङ्क grid_आकार);

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
