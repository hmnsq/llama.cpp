#pragma once

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"

#അന്തര്ഭാവ "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

// Quantization
ശൂന്യം quantize_row_q1_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q2_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q4_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q4_1(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q5_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q5_1(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q8_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q8_1(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);

ശൂന്യം quantize_row_mxfp4(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_nvfp4(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);

ശൂന്യം quantize_row_q2_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q3_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q4_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q5_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q6_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_q8_K(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);

ശൂന്യം quantize_row_tq1_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_tq2_0(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);

ശൂന്യം quantize_row_iq4_nl (സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം quantize_row_iq4_xs (സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);

// Dot product
ശൂന്യം ggml_vec_dot_q1_0_q8_0(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q2_0_q8_0(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q4_0_q8_0(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q4_1_q8_1(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q5_0_q8_0(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q5_1_q8_1(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q8_0_q8_0(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

ശൂന്യം ggml_vec_dot_mxfp4_q8_0(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_nvfp4_q8_0(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

ശൂന്യം ggml_vec_dot_q2_K_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q3_K_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q4_K_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q5_K_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q6_K_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

ശൂന്യം ggml_vec_dot_tq1_0_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_tq2_0_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

ശൂന്യം ggml_vec_dot_iq2_xxs_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq2_xs_q8_K (പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq2_s_q8_K  (പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq3_xxs_q8_K(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq1_s_q8_K  (പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq1_m_q8_K  (പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq4_nl_q8_0 (പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq4_xs_q8_K (പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq3_s_q8_K  (പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

// Generic implementation
ശൂന്യം quantize_row_q8_0_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k);
ശൂന്യം quantize_row_q8_1_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT vy, int64_t k);
ശൂന്യം quantize_row_q8_K_generic(സ്ഥിരം ദശാംശം * GGML_RESTRICT x, ശൂന്യം * GGML_RESTRICT y, int64_t k);
ശൂന്യം ggml_vec_dot_q1_0_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q2_0_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q4_0_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q4_1_q8_1_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q5_0_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q5_1_q8_1_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q8_0_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

ശൂന്യം ggml_vec_dot_mxfp4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_nvfp4_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

ശൂന്യം ggml_vec_dot_tq1_0_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_tq2_0_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

ശൂന്യം ggml_vec_dot_q2_K_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q3_K_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q4_K_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q5_K_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy,  size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_q6_K_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq2_xxs_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq2_xs_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq2_s_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq3_xxs_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq3_s_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq1_s_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq1_m_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq4_nl_q8_0_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);
ശൂന്യം ggml_vec_dot_iq4_xs_q8_K_generic(പൂർണ്ണസംഖ്യ n, ദശാംശം * GGML_RESTRICT s, size_t bs, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vx, size_t bx, സ്ഥിരം ശൂന്യം * GGML_RESTRICT vy, size_t by, പൂർണ്ണസംഖ്യ nrc);

#ifdef __cplusplus
}
#endif
