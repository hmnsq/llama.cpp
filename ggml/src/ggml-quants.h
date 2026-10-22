#ಸೂಚನೆ once

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"

// GGML internal header

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q1_0_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q2_0_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q4_0_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q4_1_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q5_0_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q5_1_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_0_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_1_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_mxfp4_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_nvfp4_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q2_K_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q3_K_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q4_K_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q5_K_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q6_K_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_K_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_tq1_0_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_tq2_0_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_iq3_xxs_ref(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_iq4_nl_ref (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_iq4_xs_ref (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_iq3_s_ref  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_iq2_s_ref  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಖಂಡ_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q1_0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q2_0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_1(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q5_0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q5_1(ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q8_0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_mxfp4(ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_nvfp4(ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q2_K(ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q3_K(ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q4_K(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q5_K(ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q6_K(ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_q8_K(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_tq1_0(ಸ್ಥಿರಾಂಕ ಖಂಡ_tq1_0 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_tq2_0(ಸ್ಥಿರಾಂಕ ಖಂಡ_tq2_0 * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);

GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq2_xxs(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq2_xs (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs  * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq2_s  (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s   * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq3_xxs(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq1_s  (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s   * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq1_m  (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m   * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq4_nl (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl  * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq4_xs (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs  * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);
GGML_API ಶೂನ್ಯ dequantize_ಅಡ್ಡಸಾಲು_iq3_s  (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s   * GGML_RESTRICT x, ದಶಮಾಂಶ * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq2_xs (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq2_s  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq3_xxs(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq1_s  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq1_m  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq4_nl (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq4_xs (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_iq3_s  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);

GGML_API size_t quantize_tq1_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_tq2_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);

GGML_API size_t quantize_q2_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q3_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q4_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q5_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q6_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q1_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q2_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q4_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q4_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q5_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q5_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_q8_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);

GGML_API size_t quantize_mxfp4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);
GGML_API size_t quantize_nvfp4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT src, ಶೂನ್ಯ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * imatrix);

GGML_API ಶೂನ್ಯ iq2xs_init_impl(ಎಣಿಕೆ ggml_type type);
GGML_API ಶೂನ್ಯ iq2xs_free_impl(ಎಣಿಕೆ ggml_type type);
GGML_API ಶೂನ್ಯ iq3xs_init_impl(ಪೂರ್ಣಾಂಕ grid_ಗಾತ್ರ);
GGML_API ಶೂನ್ಯ iq3xs_free_impl(ಪೂರ್ಣಾಂಕ grid_ಗಾತ್ರ);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
