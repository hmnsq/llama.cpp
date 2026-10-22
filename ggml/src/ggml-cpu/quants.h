#pragma once

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
बाह्य "C" {
#endif

// Quantization
शून्य quantize_row_q1_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q2_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q4_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q4_1(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q5_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q5_1(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q8_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q8_1(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_mxfp4(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_nvfp4(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_q2_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q3_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q4_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q5_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q6_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_q8_K(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_tq1_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_tq2_0(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

शून्य quantize_row_iq4_nl (स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य quantize_row_iq4_xs (स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);

// Dot product
शून्य ggml_vec_dot_q1_0_q8_0(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q2_0_q8_0(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q4_0_q8_0(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q4_1_q8_1(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q5_0_q8_0(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q5_1_q8_1(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q8_0_q8_0(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

शून्य ggml_vec_dot_mxfp4_q8_0(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_nvfp4_q8_0(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

शून्य ggml_vec_dot_q2_K_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q3_K_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q4_K_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q5_K_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q6_K_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

शून्य ggml_vec_dot_tq1_0_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_tq2_0_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

शून्य ggml_vec_dot_iq2_xxs_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq2_xs_q8_K (पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq2_s_q8_K  (पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq3_xxs_q8_K(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq1_s_q8_K  (पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq1_m_q8_K  (पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq4_nl_q8_0 (पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq4_xs_q8_K (पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq3_s_q8_K  (पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

// Generic implementation
शून्य quantize_row_q8_0_generic(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k);
शून्य quantize_row_q8_1_generic(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k);
शून्य quantize_row_q8_K_generic(स्थिरांक दशांश * GGML_RESTRICT x, शून्य * GGML_RESTRICT y, int64_t k);
शून्य ggml_vec_dot_q1_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q2_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q4_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q4_1_q8_1_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q5_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q5_1_q8_1_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q8_0_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

शून्य ggml_vec_dot_mxfp4_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_nvfp4_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

शून्य ggml_vec_dot_tq1_0_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_tq2_0_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

शून्य ggml_vec_dot_q2_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q3_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q4_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q5_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy,  माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_q6_K_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq2_xxs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq2_xs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq2_s_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq3_xxs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq3_s_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq1_s_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq1_m_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq4_nl_q8_0_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);
शून्य ggml_vec_dot_iq4_xs_q8_K_generic(पूर्णांक n, दशांश * GGML_RESTRICT s, माप_प्रकार bs, स्थिरांक शून्य * GGML_RESTRICT vx, माप_प्रकार bx, स्थिरांक शून्य * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc);

#ifdef __cplusplus
}
#endif
