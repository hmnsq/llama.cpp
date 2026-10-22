#ਹਦਾਇਤ once

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

// GGML internal header

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q1_0_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q2_0_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_0_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_1_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_0_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_1_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_0_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_1_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_mxfp4_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_nvfp4_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q2_K_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q3_K_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_K_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_K_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q6_K_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_K_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_tq1_0_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_tq2_0_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_iq3_xxs_ref(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_iq4_nl_ref (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_iq4_xs_ref (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_iq3_s_ref  (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ quantize_ਪੰਕਤੀ_iq2_s_ref  (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖੰਡ_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q1_0(ਅਚਲ ਖੰਡ_q1_0 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q2_0(ਅਚਲ ਖੰਡ_q2_0 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_0(ਅਚਲ ਖੰਡ_q4_0 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_1(ਅਚਲ ਖੰਡ_q4_1 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q5_0(ਅਚਲ ਖੰਡ_q5_0 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q5_1(ਅਚਲ ਖੰਡ_q5_1 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q8_0(ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_mxfp4(ਅਚਲ ਖੰਡ_mxfp4 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_nvfp4(ਅਚਲ ਖੰਡ_nvfp4 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q2_K(ਅਚਲ ਖੰਡ_q2_K * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q3_K(ਅਚਲ ਖੰਡ_q3_K * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q4_K(ਅਚਲ ਖੰਡ_q4_K * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q5_K(ਅਚਲ ਖੰਡ_q5_K * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q6_K(ਅਚਲ ਖੰਡ_q6_K * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_q8_K(ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_tq1_0(ਅਚਲ ਖੰਡ_tq1_0 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_tq2_0(ਅਚਲ ਖੰਡ_tq2_0 * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);

GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq2_xxs(ਅਚਲ ਖੰਡ_iq2_xxs * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq2_xs (ਅਚਲ ਖੰਡ_iq2_xs  * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq2_s  (ਅਚਲ ਖੰਡ_iq2_s   * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq3_xxs(ਅਚਲ ਖੰਡ_iq3_xxs * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq1_s  (ਅਚਲ ਖੰਡ_iq1_s   * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq1_m  (ਅਚਲ ਖੰਡ_iq1_m   * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq4_nl (ਅਚਲ ਖੰਡ_iq4_nl  * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq4_xs (ਅਚਲ ਖੰਡ_iq4_xs  * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);
GGML_API ਖਾਲੀ dequantize_ਪੰਕਤੀ_iq3_s  (ਅਚਲ ਖੰਡ_iq3_s   * GGML_RESTRICT x, ਦਸ਼ਮਲਵ * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq2_xs (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq2_s  (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq3_xxs(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq1_s  (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq1_m  (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq4_nl (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq4_xs (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_iq3_s  (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);

GGML_API size_t quantize_tq1_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_tq2_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);

GGML_API size_t quantize_q2_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q3_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q4_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q5_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q6_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q1_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q2_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q4_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q4_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q5_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q5_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_q8_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);

GGML_API size_t quantize_mxfp4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);
GGML_API size_t quantize_nvfp4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT src, ਖਾਲੀ * GGML_RESTRICT dst, int64_t nrows, int64_t n_ਪ੍ਰਤੀ_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * imatrix);

GGML_API ਖਾਲੀ iq2xs_init_impl(ਗਿਣਤੀ ggml_type type);
GGML_API ਖਾਲੀ iq2xs_free_impl(ਗਿਣਤੀ ggml_type type);
GGML_API ਖਾਲੀ iq3xs_init_impl(ਪੂਰਨ grid_ਮਾਪ);
GGML_API ਖਾਲੀ iq3xs_free_impl(ਪੂਰਨ grid_ਮਾਪ);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
