#సూచన once

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"

#అన్తర్భావ "ggml.h"

// GGML CPU internal header

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

// Quantization
శూన్య quantize_అడ్డువరుస_q1_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q2_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q4_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q4_1(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q5_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q5_1(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q8_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q8_1(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);

శూన్య quantize_అడ్డువరుస_mxfp4(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_nvfp4(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);

శూన్య quantize_అడ్డువరుస_q2_K(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q3_K(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q4_K(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q5_K(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q6_K(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_q8_K(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);

శూన్య quantize_అడ్డువరుస_tq1_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_tq2_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);

శూన్య quantize_అడ్డువరుస_iq4_nl (స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య quantize_అడ్డువరుస_iq4_xs (స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);

// Dot product
శూన్య ggml_vec_dot_q1_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q2_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q4_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q4_1_q8_1(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q5_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q5_1_q8_1(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q8_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

శూన్య ggml_vec_dot_mxfp4_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_nvfp4_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

శూన్య ggml_vec_dot_q2_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q3_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q4_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q5_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q6_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

శూన్య ggml_vec_dot_tq1_0_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_tq2_0_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

శూన్య ggml_vec_dot_iq2_xxs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq2_xs_q8_K (పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq2_s_q8_K  (పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq3_xxs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq1_s_q8_K  (పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq1_m_q8_K  (పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq4_nl_q8_0 (పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq4_xs_q8_K (పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq3_s_q8_K  (పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

// Generic implementation
శూన్య quantize_అడ్డువరుస_q8_0_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k);
శూన్య quantize_అడ్డువరుస_q8_1_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k);
శూన్య quantize_అడ్డువరుస_q8_K_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k);
శూన్య ggml_vec_dot_q1_0_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q2_0_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q4_0_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q4_1_q8_1_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q5_0_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q5_1_q8_1_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q8_0_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

శూన్య ggml_vec_dot_mxfp4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_nvfp4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

శూన్య ggml_vec_dot_tq1_0_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_tq2_0_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

శూన్య ggml_vec_dot_q2_K_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q3_K_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q4_K_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q5_K_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy,  size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_q6_K_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq2_xxs_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq2_xs_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq2_s_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq3_xxs_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq3_s_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq1_s_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq1_m_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq4_nl_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);
శూన్య ggml_vec_dot_iq4_xs_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
