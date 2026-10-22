#pragma once

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"

#ଅନ୍ତର୍ଭାଵ "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

// Quantization
ଶୂନ୍ୟ quantize_row_q1_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q2_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q4_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q4_1(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q5_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q5_1(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q8_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q8_1(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);

ଶୂନ୍ୟ quantize_row_mxfp4(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_nvfp4(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);

ଶୂନ୍ୟ quantize_row_q2_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q3_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q4_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q5_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q6_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_q8_K(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);

ଶୂନ୍ୟ quantize_row_tq1_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_tq2_0(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);

ଶୂନ୍ୟ quantize_row_iq4_nl (ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ quantize_row_iq4_xs (ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);

// Dot product
ଶୂନ୍ୟ ggml_vec_dot_q1_0_q8_0(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q2_0_q8_0(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q4_0_q8_0(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q4_1_q8_1(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q5_0_q8_0(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q5_1_q8_1(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q8_0_q8_0(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_dot_mxfp4_q8_0(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_nvfp4_q8_0(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_dot_q2_K_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q3_K_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q4_K_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q5_K_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q6_K_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_dot_tq1_0_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_tq2_0_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_dot_iq2_xxs_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq2_xs_q8_K (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq2_s_q8_K  (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq3_xxs_q8_K(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq1_s_q8_K  (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq1_m_q8_K  (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq4_nl_q8_0 (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq4_xs_q8_K (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq3_s_q8_K  (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

// Generic implementation
ଶୂନ୍ୟ quantize_row_q8_0_generic(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT vy, int64_t k);
ଶୂନ୍ୟ quantize_row_q8_1_generic(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT vy, int64_t k);
ଶୂନ୍ୟ quantize_row_q8_K_generic(ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଶୂନ୍ୟ * GGML_RESTRICT y, int64_t k);
ଶୂନ୍ୟ ggml_vec_dot_q1_0_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q2_0_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q4_0_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q4_1_q8_1_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q5_0_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q5_1_q8_1_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q8_0_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_dot_mxfp4_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_nvfp4_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_dot_tq1_0_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_tq2_0_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_dot_q2_K_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q3_K_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q4_K_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q5_K_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy,  size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_q6_K_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq2_xxs_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq2_xs_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq2_s_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq3_xxs_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq3_s_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq1_s_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq1_m_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq4_nl_q8_0_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_iq4_xs_q8_K_generic(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vx, size_t bx, ଅଚଳ ଶୂନ୍ୟ * GGML_RESTRICT vy, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

#ifdef __cplusplus
}
#endif
