#pragma once

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"

#અન્તર્ભાવ "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
બાહ્ય "C" {
#endif

// Quantization
શૂન્ય quantize_row_q1_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q2_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q4_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q4_1(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q5_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q5_1(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q8_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q8_1(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);

શૂન્ય quantize_row_mxfp4(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_nvfp4(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);

શૂન્ય quantize_row_q2_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q3_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q4_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q5_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q6_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_q8_K(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);

શૂન્ય quantize_row_tq1_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_tq2_0(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);

શૂન્ય quantize_row_iq4_nl (અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય quantize_row_iq4_xs (અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);

// Dot product
શૂન્ય ggml_vec_dot_q1_0_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q2_0_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q4_0_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q4_1_q8_1(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q5_0_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q5_1_q8_1(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q8_0_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

શૂન્ય ggml_vec_dot_mxfp4_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_nvfp4_q8_0(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

શૂન્ય ggml_vec_dot_q2_K_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q3_K_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q4_K_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q5_K_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q6_K_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

શૂન્ય ggml_vec_dot_tq1_0_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_tq2_0_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

શૂન્ય ggml_vec_dot_iq2_xxs_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq2_xs_q8_K (પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq2_s_q8_K  (પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq3_xxs_q8_K(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq1_s_q8_K  (પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq1_m_q8_K  (પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq4_nl_q8_0 (પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq4_xs_q8_K (પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq3_s_q8_K  (પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

// Generic implementation
શૂન્ય quantize_row_q8_0_generic(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k);
શૂન્ય quantize_row_q8_1_generic(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT vy, int64_t k);
શૂન્ય quantize_row_q8_K_generic(અચલ દશાંશ * GGML_RESTRICT x, શૂન્ય * GGML_RESTRICT y, int64_t k);
શૂન્ય ggml_vec_dot_q1_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q2_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q4_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q4_1_q8_1_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q5_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q5_1_q8_1_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q8_0_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

શૂન્ય ggml_vec_dot_mxfp4_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_nvfp4_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

શૂન્ય ggml_vec_dot_tq1_0_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_tq2_0_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

શૂન્ય ggml_vec_dot_q2_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q3_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q4_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q5_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy,  size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_q6_K_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq2_xxs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq2_xs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq2_s_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq3_xxs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq3_s_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq1_s_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq1_m_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq4_nl_q8_0_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);
શૂન્ય ggml_vec_dot_iq4_xs_q8_K_generic(પૂર્ણાંક n, દશાંશ * GGML_RESTRICT s, size_t bs, અચલ શૂન્ય * GGML_RESTRICT vx, size_t bx, અચલ શૂન્ય * GGML_RESTRICT vy, size_t by, પૂર્ણાંક nrc);

#ifdef __cplusplus
}
#endif
