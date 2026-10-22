#ಸೂಚನೆ once

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "ggml.h"

// GGML CPU internal header

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

// Quantization
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q1_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q2_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q4_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q4_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q5_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q5_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_mxfp4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_nvfp4(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q2_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q3_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q4_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q5_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q6_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_K(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_tq1_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_tq2_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_iq4_nl (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_iq4_xs (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);

// Dot product
ಶೂನ್ಯ ggml_vec_dot_q1_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q2_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_1_q8_1(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_1_q8_1(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q8_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_nvfp4_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_tq2_0_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K (ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K  (ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K  (ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K  (ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0 (ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K (ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K  (ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

// Generic implementation
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_0_generic(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_1_generic(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k);
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_K_generic(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT y, int64_t k);
ಶೂನ್ಯ ggml_vec_dot_q1_0_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q2_0_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_0_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_1_q8_1_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_0_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_1_q8_1_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q8_0_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_nvfp4_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_tq1_0_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_tq2_0_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q5_K_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy,  size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq1_m_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);
ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K_generic(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
