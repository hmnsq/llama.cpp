#ਹਦਾਇਤ once

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

// GGML CPU internal header

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

// Quantization
ਖਾਲੀ quantize_ਪੰਕਤੀ_q1_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q2_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);

ਖਾਲੀ quantize_ਪੰਕਤੀ_mxfp4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_nvfp4(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);

ਖਾਲੀ quantize_ਪੰਕਤੀ_q2_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q3_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q4_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q5_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q6_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_K(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);

ਖਾਲੀ quantize_ਪੰਕਤੀ_tq1_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_tq2_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);

ਖਾਲੀ quantize_ਪੰਕਤੀ_iq4_nl (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_iq4_xs (ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);

// Dot product
ਖਾਲੀ ggml_vec_dot_q1_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q2_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q4_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q4_1_q8_1(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q5_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q5_1_q8_1(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q8_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

ਖਾਲੀ ggml_vec_dot_mxfp4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_nvfp4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

ਖਾਲੀ ggml_vec_dot_q2_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q3_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q4_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q5_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q6_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

ਖਾਲੀ ggml_vec_dot_tq1_0_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_tq2_0_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

ਖਾਲੀ ggml_vec_dot_iq2_xxs_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq2_xs_q8_K (ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq2_s_q8_K  (ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq3_xxs_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq1_s_q8_K  (ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq1_m_q8_K  (ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq4_nl_q8_0 (ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq4_xs_q8_K (ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq3_s_q8_K  (ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

// Generic implementation
ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_0_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_1_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k);
ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_K_generic(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT y, int64_t k);
ਖਾਲੀ ggml_vec_dot_q1_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q2_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q4_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q4_1_q8_1_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q5_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q5_1_q8_1_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q8_0_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

ਖਾਲੀ ggml_vec_dot_mxfp4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_nvfp4_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

ਖਾਲੀ ggml_vec_dot_tq1_0_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_tq2_0_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

ਖਾਲੀ ggml_vec_dot_q2_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q3_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q4_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q5_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,  size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_q6_K_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq2_xxs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq2_xs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq2_s_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq3_xxs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq3_s_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq1_s_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq1_m_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq4_nl_q8_0_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);
ਖਾਲੀ ggml_vec_dot_iq4_xs_q8_K_generic(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
