#ہدایت once

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"

#ضمن "ggml.h"

// GGML CPU internal header

#جے_تعریف_شدہ __cplusplus
خارجي "C" {
#جے_انت

// Quantization
فراغ quantize_صف_q1_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q2_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q4_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q4_1(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q5_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q5_1(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q8_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q8_1(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);

فراغ quantize_صف_mxfp4(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_nvfp4(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);

فراغ quantize_صف_q2_K(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q3_K(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q4_K(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q5_K(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q6_K(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_q8_K(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);

فراغ quantize_صف_tq1_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_tq2_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);

فراغ quantize_صف_iq4_nl (ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ quantize_صف_iq4_xs (ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);

// Dot product
فراغ ggml_vec_dot_q1_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q2_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q4_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q4_1_q8_1(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q5_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q5_1_q8_1(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q8_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

فراغ ggml_vec_dot_mxfp4_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_nvfp4_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

فراغ ggml_vec_dot_q2_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q3_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q4_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q5_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q6_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

فراغ ggml_vec_dot_tq1_0_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_tq2_0_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

فراغ ggml_vec_dot_iq2_xxs_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq2_xs_q8_K (صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq2_s_q8_K  (صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq3_xxs_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq1_s_q8_K  (صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq1_m_q8_K  (صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq4_nl_q8_0 (صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq4_xs_q8_K (صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq3_s_q8_K  (صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

// Generic implementation
فراغ quantize_صف_q8_0_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k);
فراغ quantize_صف_q8_1_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k);
فراغ quantize_صف_q8_K_generic(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT y, int64_t k);
فراغ ggml_vec_dot_q1_0_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q2_0_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q4_0_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q4_1_q8_1_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q5_0_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q5_1_q8_1_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q8_0_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

فراغ ggml_vec_dot_mxfp4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_nvfp4_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

فراغ ggml_vec_dot_tq1_0_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_tq2_0_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

فراغ ggml_vec_dot_q2_K_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q3_K_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q4_K_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_q5_K_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy,  size_t by, صحيح nrc);
فراغ ggml_vec_dot_q6_K_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq2_xxs_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq2_xs_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq2_s_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq3_xxs_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq3_s_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq1_s_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq1_m_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq4_nl_q8_0_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);
فراغ ggml_vec_dot_iq4_xs_q8_K_generic(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc);

#جے_تعریف_شدہ __cplusplus
}
#جے_انت
