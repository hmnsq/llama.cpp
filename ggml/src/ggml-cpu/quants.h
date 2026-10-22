#pragma once

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"

#অন্তর্ভাব "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
বাহ্য "C" {
#endif

// Quantization
শূন্য quantize_row_q1_0(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q2_0(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q4_0(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q4_1(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q5_0(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q5_1(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q8_0(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q8_1(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);

শূন্য quantize_row_mxfp4(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_nvfp4(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);

শূন্য quantize_row_q2_K(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q3_K(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q4_K(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q5_K(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q6_K(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_q8_K(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);

শূন্য quantize_row_tq1_0(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_tq2_0(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);

শূন্য quantize_row_iq4_nl (ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য quantize_row_iq4_xs (ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);

// Dot product
শূন্য ggml_vec_dot_q1_0_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q2_0_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q4_0_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q4_1_q8_1(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q5_0_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q5_1_q8_1(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q8_0_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

শূন্য ggml_vec_dot_mxfp4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_nvfp4_q8_0(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

শূন্য ggml_vec_dot_q2_K_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q3_K_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q4_K_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q5_K_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q6_K_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

শূন্য ggml_vec_dot_tq1_0_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_tq2_0_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

শূন্য ggml_vec_dot_iq2_xxs_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq2_xs_q8_K (পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq2_s_q8_K  (পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq3_xxs_q8_K(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq1_s_q8_K  (পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq1_m_q8_K  (পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq4_nl_q8_0 (পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq4_xs_q8_K (পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq3_s_q8_K  (পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

// Generic implementation
শূন্য quantize_row_q8_0_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য quantize_row_q8_1_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT vy, int64_t k);
শূন্য quantize_row_q8_K_generic(ধ্রুব ভিন্ন * GGML_RESTRICT x, শূন্য * GGML_RESTRICT y, int64_t k);
শূন্য ggml_vec_dot_q1_0_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q2_0_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q4_0_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q4_1_q8_1_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q5_0_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q5_1_q8_1_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q8_0_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

শূন্য ggml_vec_dot_mxfp4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_nvfp4_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

শূন্য ggml_vec_dot_tq1_0_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_tq2_0_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

শূন্য ggml_vec_dot_q2_K_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q3_K_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q4_K_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q5_K_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy,  size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_q6_K_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq2_xxs_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq2_xs_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq2_s_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq3_xxs_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq3_s_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq1_s_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq1_m_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq4_nl_q8_0_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);
শূন্য ggml_vec_dot_iq4_xs_q8_K_generic(পূর্ণাঙ্ক n, ভিন্ন * GGML_RESTRICT s, size_t bs, ধ্রুব শূন্য * GGML_RESTRICT vx, size_t bx, ধ্রুব শূন্য * GGML_RESTRICT vy, size_t by, পূর্ণাঙ্ক nrc);

#ifdef __cplusplus
}
#endif
