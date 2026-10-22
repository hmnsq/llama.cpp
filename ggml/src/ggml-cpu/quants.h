#ہدایت once

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"

#شامل "ggml.h"

// GGML CPU internal header

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

// Quantization
خالی quantize_صف_q1_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q2_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q4_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q4_1(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q5_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q5_1(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q8_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q8_1(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);

خالی quantize_صف_mxfp4(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_nvfp4(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);

خالی quantize_صف_q2_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q3_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q4_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q5_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q6_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_q8_K(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);

خالی quantize_صف_tq1_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_tq2_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);

خالی quantize_صف_iq4_nl (ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی quantize_صف_iq4_xs (ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);

// Dot product
خالی ggml_vec_dot_q1_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q2_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q4_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q4_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q5_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q5_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q8_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

خالی ggml_vec_dot_mxfp4_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_nvfp4_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

خالی ggml_vec_dot_q2_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q3_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q4_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q5_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q6_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

خالی ggml_vec_dot_tq1_0_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_tq2_0_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

خالی ggml_vec_dot_iq2_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq2_xs_q8_K (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq2_s_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq3_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq1_s_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq1_m_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq4_nl_q8_0 (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq4_xs_q8_K (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq3_s_q8_K  (صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

// Generic implementation
خالی quantize_صف_q8_0_generic(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT vy, int64_t k);
خالی quantize_صف_q8_1_generic(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT vy, int64_t k);
خالی quantize_صف_q8_K_generic(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT y, int64_t k);
خالی ggml_vec_dot_q1_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q2_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q4_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q4_1_q8_1_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q5_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q5_1_q8_1_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q8_0_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

خالی ggml_vec_dot_mxfp4_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_nvfp4_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

خالی ggml_vec_dot_tq1_0_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_tq2_0_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

خالی ggml_vec_dot_q2_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q3_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q4_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_q5_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy,  size_t by, صحیح nrc);
خالی ggml_vec_dot_q6_K_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq2_xxs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq2_xs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq2_s_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq3_xxs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq3_s_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq1_s_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq1_m_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq4_nl_q8_0_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);
خالی ggml_vec_dot_iq4_xs_q8_K_generic(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc);

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
