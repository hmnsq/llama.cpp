#pragma once

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

// Quantization
ಶೂನ್ಯ quantize_row_q1_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q2_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q4_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q4_1(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q5_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q5_1(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q8_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q8_1(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_row_mxfp4(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_nvfp4(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_row_q2_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q3_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q4_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q5_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q6_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_q8_K(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_row_tq1_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_tq2_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_row_iq4_nl (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_row_iq4_xs (ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

// Dot product
ಶೂನ್ಯ ggml_vec_dot_q1_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q2_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_1_q8_1(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_1_q8_1(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q8_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_nvfp4_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_tq2_0_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K (ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K  (ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K  (ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K  (ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0 (ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K (ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K  (ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

// Generic implementation
ಶೂನ್ಯ quantize_row_q8_0_generic(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k);
ಶೂನ್ಯ quantize_row_q8_1_generic(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k);
ಶೂನ್ಯ quantize_row_q8_K_generic(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ ggml_vec_dot_q1_0_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q2_0_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_0_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_1_q8_1_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_0_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_1_q8_1_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q8_0_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_nvfp4_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_tq2_0_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_K_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy,  size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K_generic(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc);

#ifdef __cplusplus
}
#endif
