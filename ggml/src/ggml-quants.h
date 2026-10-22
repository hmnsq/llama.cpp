#رهنمود once

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"

#ضمن "ggml.h"

// GGML internal header

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API فراغ quantize_ردیف_q1_0_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q2_0_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q2_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q4_0_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q4_1_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q5_0_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q5_1_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q8_0_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q8_1_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_ردیف_mxfp4_ref(ثابت عائم * GGML_RESTRICT x, بلوک_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_nvfp4_ref(ثابت عائم * GGML_RESTRICT x, بلوک_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_ردیف_q2_K_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q3_K_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q4_K_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q5_K_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q6_K_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_q8_K_ref(ثابت عائم * GGML_RESTRICT x, بلوک_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_ردیف_tq1_0_ref(ثابت عائم * GGML_RESTRICT x, بلوک_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_tq2_0_ref(ثابت عائم * GGML_RESTRICT x, بلوک_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API فراغ quantize_ردیف_iq3_xxs_ref(ثابت عائم * GGML_RESTRICT x, بلوک_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_iq4_nl_ref (ثابت عائم * GGML_RESTRICT x, بلوک_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_iq4_xs_ref (ثابت عائم * GGML_RESTRICT x, بلوک_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_iq3_s_ref  (ثابت عائم * GGML_RESTRICT x, بلوک_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API فراغ quantize_ردیف_iq2_s_ref  (ثابت عائم * GGML_RESTRICT x, بلوک_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API فراغ dequantize_ردیف_q1_0(ثابت بلوک_q1_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q2_0(ثابت بلوک_q2_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q4_0(ثابت بلوک_q4_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q4_1(ثابت بلوک_q4_1 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q5_0(ثابت بلوک_q5_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q5_1(ثابت بلوک_q5_1 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q8_0(ثابت بلوک_q8_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_ردیف_mxfp4(ثابت بلوک_mxfp4 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_nvfp4(ثابت بلوک_nvfp4 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_ردیف_q2_K(ثابت بلوک_q2_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q3_K(ثابت بلوک_q3_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q4_K(ثابت بلوک_q4_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q5_K(ثابت بلوک_q5_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q6_K(ثابت بلوک_q6_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_q8_K(ثابت بلوک_q8_K * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_ردیف_tq1_0(ثابت بلوک_tq1_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_tq2_0(ثابت بلوک_tq2_0 * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

GGML_API فراغ dequantize_ردیف_iq2_xxs(ثابت بلوک_iq2_xxs * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq2_xs (ثابت بلوک_iq2_xs  * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq2_s  (ثابت بلوک_iq2_s   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq3_xxs(ثابت بلوک_iq3_xxs * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq1_s  (ثابت بلوک_iq1_s   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq1_m  (ثابت بلوک_iq1_m   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq4_nl (ثابت بلوک_iq4_nl  * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq4_xs (ثابت بلوک_iq4_xs  * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);
GGML_API فراغ dequantize_ردیف_iq3_s  (ثابت بلوک_iq3_s   * GGML_RESTRICT x, عائم * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq2_xs (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq2_s  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq3_xxs(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq1_s  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq1_m  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq4_nl (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq4_xs (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_iq3_s  (ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);

GGML_API size_t quantize_tq1_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_tq2_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);

GGML_API size_t quantize_q2_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q3_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q4_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q5_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q6_K(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q1_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q2_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q4_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q4_1(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q5_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q5_1(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_q8_0(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);

GGML_API size_t quantize_mxfp4(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);
GGML_API size_t quantize_nvfp4(ثابت عائم * GGML_RESTRICT src, فراغ * GGML_RESTRICT dst, int64_t nrows, int64_t n_در_هر_ردیف, ثابت عائم * imatrix);

GGML_API فراغ iq2xs_init_impl(تعداد ggml_type type);
GGML_API فراغ iq2xs_free_impl(تعداد ggml_type type);
GGML_API فراغ iq3xs_init_impl(صحيح grid_اندازه);
GGML_API فراغ iq3xs_free_impl(صحيح grid_اندازه);

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
