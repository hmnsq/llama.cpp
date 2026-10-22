#pragma once

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
बाह्य "C" {
#endif

// Quantization
शून्य quantize_row_q1_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q2_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q4_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q4_1(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q5_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q5_1(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q8_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q8_1(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_mxfp4(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_nvfp4(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_q2_K(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q3_K(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q4_K(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q5_K(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q6_K(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q8_K(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_tq1_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_tq2_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_iq4_nl (ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_iq4_xs (ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

// Dot product
शून्य ggml_vec_dot_q1_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q2_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q4_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q4_1_q8_1(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q5_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q5_1_q8_1(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q8_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

शून्य ggml_vec_dot_mxfp4_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_nvfp4_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

शून्य ggml_vec_dot_q2_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q3_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q4_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q5_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q6_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

शून्य ggml_vec_dot_tq1_0_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_tq2_0_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

शून्य ggml_vec_dot_iq2_xxs_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq2_xs_q8_K (पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq2_s_q8_K  (पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq3_xxs_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq1_s_q8_K  (पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq1_m_q8_K  (पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq4_nl_q8_0 (पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq4_xs_q8_K (पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq3_s_q8_K  (पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

// Generic implementation
शून्य quantize_row_q8_0_generic(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k);
शून्य quantize_row_q8_1_generic(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k);
शून्य quantize_row_q8_K_generic(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य ggml_vec_dot_q1_0_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q2_0_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q4_0_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q4_1_q8_1_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q5_0_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q5_1_q8_1_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q8_0_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

शून्य ggml_vec_dot_mxfp4_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_nvfp4_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

शून्य ggml_vec_dot_tq1_0_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_tq2_0_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

शून्य ggml_vec_dot_q2_K_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q3_K_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q4_K_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q5_K_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy,  माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_q6_K_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq2_xxs_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq2_xs_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq2_s_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq3_xxs_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq3_s_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq1_s_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq1_m_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq4_nl_q8_0_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);
शून्य ggml_vec_dot_iq4_xs_q8_K_generic(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, माप_प्रकार bs, ध्रुव शून्य * GGML_RESTRICT vx, माप_प्रकार bx, ध्रुव शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णाङ्क nrc);

#ifdef __cplusplus
}
#endif
