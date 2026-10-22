#pragma once

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"

#അന്തര്ഭാവ "ggml.h"

// GGML internal header

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API ശൂന്യം quantize_row_q1_0_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q2_0_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q4_0_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q4_1_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q5_0_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q5_1_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q8_0_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q8_1_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം quantize_row_mxfp4_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_nvfp4_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം quantize_row_q2_K_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q3_K_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q4_K_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q5_K_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q6_K_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_q8_K_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം quantize_row_tq1_0_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_tq2_0_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം quantize_row_iq3_xxs_ref(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_iq4_nl_ref (സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_iq4_xs_ref (സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_iq3_s_ref  (സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം quantize_row_iq2_s_ref  (സ്ഥിരം ദശാംശം * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API ശൂന്യം dequantize_row_q1_0(സ്ഥിരം block_q1_0 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q2_0(സ്ഥിരം block_q2_0 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q4_0(സ്ഥിരം block_q4_0 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q4_1(സ്ഥിരം block_q4_1 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q5_0(സ്ഥിരം block_q5_0 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q5_1(സ്ഥിരം block_q5_1 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q8_0(സ്ഥിരം block_q8_0 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം dequantize_row_mxfp4(സ്ഥിരം block_mxfp4 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_nvfp4(സ്ഥിരം block_nvfp4 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം dequantize_row_q2_K(സ്ഥിരം block_q2_K * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q3_K(സ്ഥിരം block_q3_K * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q4_K(സ്ഥിരം block_q4_K * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q5_K(സ്ഥിരം block_q5_K * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q6_K(സ്ഥിരം block_q6_K * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_q8_K(സ്ഥിരം block_q8_K * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം dequantize_row_tq1_0(സ്ഥിരം block_tq1_0 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_tq2_0(സ്ഥിരം block_tq2_0 * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);

GGML_API ശൂന്യം dequantize_row_iq2_xxs(സ്ഥിരം block_iq2_xxs * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq2_xs (സ്ഥിരം block_iq2_xs  * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq2_s  (സ്ഥിരം block_iq2_s   * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq3_xxs(സ്ഥിരം block_iq3_xxs * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq1_s  (സ്ഥിരം block_iq1_s   * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq1_m  (സ്ഥിരം block_iq1_m   * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq4_nl (സ്ഥിരം block_iq4_nl  * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq4_xs (സ്ഥിരം block_iq4_xs  * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);
GGML_API ശൂന്യം dequantize_row_iq3_s  (സ്ഥിരം block_iq3_s   * GGML_RESTRICT x, ദശാംശം * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq2_xs (സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq2_s  (സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq3_xxs(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq1_s  (സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq1_m  (സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq4_nl (സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq4_xs (സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_iq3_s  (സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);

GGML_API size_t quantize_tq1_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_tq2_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);

GGML_API size_t quantize_q2_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q3_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q4_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q5_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q6_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q1_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q2_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q4_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q4_1(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q5_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q5_1(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_q8_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);

GGML_API size_t quantize_mxfp4(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);
GGML_API size_t quantize_nvfp4(സ്ഥിരം ദശാംശം * GGML_RESTRICT src, ശൂന്യം * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, സ്ഥിരം ദശാംശം * imatrix);

GGML_API ശൂന്യം iq2xs_init_impl(എണ്ണൽ ggml_type type);
GGML_API ശൂന്യം iq2xs_free_impl(എണ്ണൽ ggml_type type);
GGML_API ശൂന്യം iq3xs_init_impl(പൂർണ്ണസംഖ്യ grid_size);
GGML_API ശൂന്യം iq3xs_free_impl(പൂർണ്ണസംഖ്യ grid_size);

#ifdef __cplusplus
}
#endif
