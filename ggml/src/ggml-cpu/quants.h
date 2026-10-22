#pragma once

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"

#அந்தர்பாவ "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
வெளி "C" {
#endif

// Quantization
வெறுமை quantize_row_q1_0(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q2_0(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q4_0(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q4_1(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q5_0(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q5_1(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q8_0(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q8_1(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);

வெறுமை quantize_row_mxfp4(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_nvfp4(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);

வெறுமை quantize_row_q2_K(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q3_K(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q4_K(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q5_K(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q6_K(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_q8_K(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);

வெறுமை quantize_row_tq1_0(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_tq2_0(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);

வெறுமை quantize_row_iq4_nl (மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை quantize_row_iq4_xs (மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);

// Dot product
வெறுமை ggml_vec_dot_q1_0_q8_0(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q2_0_q8_0(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q4_0_q8_0(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q4_1_q8_1(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q5_0_q8_0(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q5_1_q8_1(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q8_0_q8_0(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

வெறுமை ggml_vec_dot_mxfp4_q8_0(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_nvfp4_q8_0(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

வெறுமை ggml_vec_dot_q2_K_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q3_K_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q4_K_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q5_K_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q6_K_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

வெறுமை ggml_vec_dot_tq1_0_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_tq2_0_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

வெறுமை ggml_vec_dot_iq2_xxs_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq2_xs_q8_K (முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq2_s_q8_K  (முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq3_xxs_q8_K(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq1_s_q8_K  (முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq1_m_q8_K  (முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq4_nl_q8_0 (முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq4_xs_q8_K (முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq3_s_q8_K  (முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

// Generic implementation
வெறுமை quantize_row_q8_0_generic(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT vy, int64_t k);
வெறுமை quantize_row_q8_1_generic(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT vy, int64_t k);
வெறுமை quantize_row_q8_K_generic(மாறிலி தசம * GGML_RESTRICT x, வெறுமை * GGML_RESTRICT y, int64_t k);
வெறுமை ggml_vec_dot_q1_0_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q2_0_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q4_0_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q4_1_q8_1_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q5_0_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q5_1_q8_1_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q8_0_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

வெறுமை ggml_vec_dot_mxfp4_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_nvfp4_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

வெறுமை ggml_vec_dot_tq1_0_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_tq2_0_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

வெறுமை ggml_vec_dot_q2_K_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q3_K_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q4_K_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q5_K_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy,  size_t by, முழு nrc);
வெறுமை ggml_vec_dot_q6_K_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq2_xxs_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq2_xs_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq2_s_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq3_xxs_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq3_s_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq1_s_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq1_m_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq4_nl_q8_0_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);
வெறுமை ggml_vec_dot_iq4_xs_q8_K_generic(முழு n, தசம * GGML_RESTRICT s, size_t bs, மாறிலி வெறுமை * GGML_RESTRICT vx, size_t bx, மாறிலி வெறுமை * GGML_RESTRICT vy, size_t by, முழு nrc);

#ifdef __cplusplus
}
#endif
