#సూచన once

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"

#అన్తర్భావ "ggml.h"

// GGML internal header

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API శూన్య quantize_అడ్డువరుస_q1_0_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q2_0_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q4_0_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q4_1_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q5_0_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q5_1_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q8_0_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q8_1_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య quantize_అడ్డువరుస_mxfp4_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_nvfp4_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య quantize_అడ్డువరుస_q2_K_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q3_K_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q4_K_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q5_K_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q6_K_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_q8_K_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య quantize_అడ్డువరుస_tq1_0_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_tq2_0_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య quantize_అడ్డువరుస_iq3_xxs_ref(స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_iq4_nl_ref (స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_iq4_xs_ref (స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_iq3_s_ref  (స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య quantize_అడ్డువరుస_iq2_s_ref  (స్థిరరాశి దశాంశం * GGML_RESTRICT x, ఖండం_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API శూన్య dequantize_అడ్డువరుస_q1_0(స్థిరరాశి ఖండం_q1_0 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q2_0(స్థిరరాశి ఖండం_q2_0 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q4_0(స్థిరరాశి ఖండం_q4_0 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q4_1(స్థిరరాశి ఖండం_q4_1 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q5_0(స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q5_1(స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q8_0(స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య dequantize_అడ్డువరుస_mxfp4(స్థిరరాశి ఖండం_mxfp4 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_nvfp4(స్థిరరాశి ఖండం_nvfp4 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య dequantize_అడ్డువరుస_q2_K(స్థిరరాశి ఖండం_q2_K * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q3_K(స్థిరరాశి ఖండం_q3_K * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q4_K(స్థిరరాశి ఖండం_q4_K * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q5_K(స్థిరరాశి ఖండం_q5_K * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q6_K(స్థిరరాశి ఖండం_q6_K * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_q8_K(స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య dequantize_అడ్డువరుస_tq1_0(స్థిరరాశి ఖండం_tq1_0 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_tq2_0(స్థిరరాశి ఖండం_tq2_0 * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);

GGML_API శూన్య dequantize_అడ్డువరుస_iq2_xxs(స్థిరరాశి ఖండం_iq2_xxs * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq2_xs (స్థిరరాశి ఖండం_iq2_xs  * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq2_s  (స్థిరరాశి ఖండం_iq2_s   * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq3_xxs(స్థిరరాశి ఖండం_iq3_xxs * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq1_s  (స్థిరరాశి ఖండం_iq1_s   * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq1_m  (స్థిరరాశి ఖండం_iq1_m   * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq4_nl (స్థిరరాశి ఖండం_iq4_nl  * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq4_xs (స్థిరరాశి ఖండం_iq4_xs  * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);
GGML_API శూన్య dequantize_అడ్డువరుస_iq3_s  (స్థిరరాశి ఖండం_iq3_s   * GGML_RESTRICT x, దశాంశం * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq2_xs (స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq2_s  (స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq3_xxs(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq1_s  (స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq1_m  (స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq4_nl (స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq4_xs (స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_iq3_s  (స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);

GGML_API size_t quantize_tq1_0(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_tq2_0(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);

GGML_API size_t quantize_q2_K(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q3_K(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q4_K(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q5_K(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q6_K(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q1_0(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q2_0(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q4_0(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q4_1(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q5_0(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q5_1(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_q8_0(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);

GGML_API size_t quantize_mxfp4(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);
GGML_API size_t quantize_nvfp4(స్థిరరాశి దశాంశం * GGML_RESTRICT src, శూన్య * GGML_RESTRICT dst, int64_t nrows, int64_t n_చొప్పున_అడ్డువరుస, స్థిరరాశి దశాంశం * imatrix);

GGML_API శూన్య iq2xs_init_impl(గణన ggml_type type);
GGML_API శూన్య iq2xs_free_impl(గణన ggml_type type);
GGML_API శూన్య iq3xs_init_impl(పూర్ణాంకం grid_కొలత);
GGML_API శూన్య iq3xs_free_impl(పూర్ణాంకం grid_కొలత);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
