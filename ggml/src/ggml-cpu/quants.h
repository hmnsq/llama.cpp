#pragma once

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"

#شامل "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
بیرونی "C" {
#endif

// Quantization
تهی quantize_row_q1_0(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q2_0(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q4_0(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q4_1(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q5_0(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q5_1(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q8_0(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q8_1(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);

تهی quantize_row_mxfp4(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_nvfp4(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);

تهی quantize_row_q2_K(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q3_K(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q4_K(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q5_K(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q6_K(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_q8_K(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);

تهی quantize_row_tq1_0(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_tq2_0(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);

تهی quantize_row_iq4_nl (ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی quantize_row_iq4_xs (ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);

// Dot product
تهی ggml_vec_dot_q1_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q2_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q4_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q4_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q5_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q5_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q8_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

تهی ggml_vec_dot_mxfp4_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_nvfp4_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

تهی ggml_vec_dot_q2_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q3_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q4_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q5_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q6_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

تهی ggml_vec_dot_tq1_0_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_tq2_0_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

تهی ggml_vec_dot_iq2_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq2_xs_q8_K (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq2_s_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq3_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq1_s_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq1_m_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq4_nl_q8_0 (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq4_xs_q8_K (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq3_s_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

// Generic implementation
تهی quantize_row_q8_0_generic(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT vy, int64_t k);
تهی quantize_row_q8_1_generic(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT vy, int64_t k);
تهی quantize_row_q8_K_generic(ثابت اعشاری * GGML_RESTRICT x, تهی * GGML_RESTRICT y, int64_t k);
تهی ggml_vec_dot_q1_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q2_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q4_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q4_1_q8_1_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q5_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q5_1_q8_1_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q8_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

تهی ggml_vec_dot_mxfp4_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_nvfp4_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

تهی ggml_vec_dot_tq1_0_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_tq2_0_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

تهی ggml_vec_dot_q2_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q3_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q4_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_q5_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy,  size_t by, صحیح nrc);
تهی ggml_vec_dot_q6_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq2_xxs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq2_xs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq2_s_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq3_xxs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq3_s_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq1_s_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq1_m_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq4_nl_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);
تهی ggml_vec_dot_iq4_xs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت تهی * GGML_RESTRICT vx, size_t bx, ثابت تهی * GGML_RESTRICT vy, size_t by, صحیح nrc);

#ifdef __cplusplus
}
#endif
