#pragma once

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"

#અન્તર્ભાવ "ggml.h"

// GGML internal header

#ifdef __cplusplus
બાહ્ય "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API શૂન્ય quantize_row_q1_0_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q2_0_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q4_0_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q4_1_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q5_0_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q5_1_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q8_0_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q8_1_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય quantize_row_mxfp4_ref(અચલ દશાંશ * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_nvfp4_ref(અચલ દશાંશ * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય quantize_row_q2_K_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q3_K_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q4_K_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q5_K_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q6_K_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_q8_K_ref(અચલ દશાંશ * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય quantize_row_tq1_0_ref(અચલ દશાંશ * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_tq2_0_ref(અચલ દશાંશ * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય quantize_row_iq3_xxs_ref(અચલ દશાંશ * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_iq4_nl_ref (અચલ દશાંશ * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_iq4_xs_ref (અચલ દશાંશ * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_iq3_s_ref  (અચલ દશાંશ * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય quantize_row_iq2_s_ref  (અચલ દશાંશ * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API શૂન્ય dequantize_row_q1_0(અચલ block_q1_0 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q2_0(અચલ block_q2_0 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q4_0(અચલ block_q4_0 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q4_1(અચલ block_q4_1 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q5_0(અચલ block_q5_0 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q5_1(અચલ block_q5_1 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q8_0(અચલ block_q8_0 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય dequantize_row_mxfp4(અચલ block_mxfp4 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_nvfp4(અચલ block_nvfp4 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય dequantize_row_q2_K(અચલ block_q2_K * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q3_K(અચલ block_q3_K * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q4_K(અચલ block_q4_K * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q5_K(અચલ block_q5_K * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q6_K(અચલ block_q6_K * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_q8_K(અચલ block_q8_K * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય dequantize_row_tq1_0(અચલ block_tq1_0 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_tq2_0(અચલ block_tq2_0 * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);

GGML_API શૂન્ય dequantize_row_iq2_xxs(અચલ block_iq2_xxs * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq2_xs (અચલ block_iq2_xs  * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq2_s  (અચલ block_iq2_s   * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq3_xxs(અચલ block_iq3_xxs * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq1_s  (અચલ block_iq1_s   * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq1_m  (અચલ block_iq1_m   * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq4_nl (અચલ block_iq4_nl  * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq4_xs (અચલ block_iq4_xs  * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);
GGML_API શૂન્ય dequantize_row_iq3_s  (અચલ block_iq3_s   * GGML_RESTRICT x, દશાંશ * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq2_xs (અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq2_s  (અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq3_xxs(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq1_s  (અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq1_m  (અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq4_nl (અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq4_xs (અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_iq3_s  (અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);

GGML_API size_t quantize_tq1_0(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_tq2_0(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);

GGML_API size_t quantize_q2_K(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q3_K(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q4_K(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q5_K(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q6_K(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q1_0(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q2_0(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q4_0(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q4_1(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q5_0(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q5_1(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_q8_0(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);

GGML_API size_t quantize_mxfp4(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);
GGML_API size_t quantize_nvfp4(અચલ દશાંશ * GGML_RESTRICT src, શૂન્ય * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, અચલ દશાંશ * imatrix);

GGML_API શૂન્ય iq2xs_init_impl(ગણના ggml_type type);
GGML_API શૂન્ય iq2xs_free_impl(ગણના ggml_type type);
GGML_API શૂન્ય iq3xs_init_impl(પૂર્ણાંક grid_size);
GGML_API શૂન્ય iq3xs_free_impl(પૂર્ણાંક grid_size);

#ifdef __cplusplus
}
#endif
