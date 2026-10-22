#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_SIGMOID_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_SIGMOID_H

#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hvx-inverse.h"
#ಅನ್ತರ್ಭಾವ "hvx-exp.h"

#ವ್ಯಾಖ್ಯಾನಿಸು FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#ವ್ಯಾಖ್ಯಾನಿಸು FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#ವ್ಯಾಖ್ಯಾನಿಸು FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#ವ್ಯಾಖ್ಯಾನಿಸು FAST_SIGMOID_C3    (0x3f000000)  // 0.5

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_fast_sigmoid_f32(HVX_ಸದಿಶ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_ಸದಿಶ ಒಳ_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_ಸದಿಶ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(ಒಳ_int));
    HVX_ಸದಿಶ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_ಸದಿಶ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_ಸದಿಶ v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_ಸದಿಶ v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(ಒಳ_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, ಒಳ_int, 24);

    HVX_ಸದಿಶ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_ಸದಿಶ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_ಸದಿಶ res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(res);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_fast_sigmoid_f32_guard(HVX_ಸದಿಶ v,
                                                         HVX_ಸದಿಶ one,
                                                         HVX_ಸದಿಶ max_exp,
                                                         HVX_ಸದಿಶ min_exp) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_ಸದಿಶ ಹೊರಹೋಗು = hvx_vec_fast_sigmoid_f32(v);
    ಹೊರಹೋಗು            = Q6_V_vmux_QVV(pred_max, ಹೊರಹೋಗು, one);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(pred_min, ಹೊರಹೋಗು, Q6_V_vzero());
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_tanh_f32(HVX_ಸದಿಶ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_ಸದಿಶ two = hvx_vec_splat_f32(2.0f);
    HVX_ಸದಿಶ one = hvx_vec_splat_f32(1.0f);
    HVX_ಸದಿಶ x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_ಸದಿಶ max_exp = hvx_vec_splat_f32(87.f);
    HVX_ಸದಿಶ min_exp = hvx_vec_splat_f32(-87.f);

    HVX_ಸದಿಶ sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_ಸದಿಶ res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(res);
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_sigmoid_loop_body(dst_type, src_type, vec_store)                                  \
    ಮಾಡು {                                                                                      \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                          \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                          \
                                                                                              \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one     = hvx_vec_splat_f32(1.f);                                    \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_exp = hvx_vec_splat_f32(87.f);                                   \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_exp = hvx_vec_splat_f32(-87.f);                                  \
                                                                                              \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);                                            \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                                        \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                                        \
                                                                                              \
        uint32_t i = 0;                                                                       \
                                                                                              \
        _Pragma("unroll(4)")                                                                  \
        ಗಾಗಿ (; i < nvec; i++) {                                                               \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp);        \
        }                                                                                     \
        ಆದರೆ (nloe) {                                                                           \
             HVX_ಸದಿಶ tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), tmp);                         \
        }                                                                                     \
    } ಆಗ(0)

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_tanh_loop_body(dst_type, src_type, vec_store)             \
    ಮಾಡು {                                                              \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                  \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                  \
                                                                      \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);                    \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                \
                                                                      \
        uint32_t i = 0;                                               \
                                                                      \
        _Pragma("unroll(4)")                                          \
        ಗಾಗಿ (; i < nvec; i++) {                                       \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);                     \
        }                                                             \
        ಆದರೆ (nloe) {                                                   \
             HVX_ಸದಿಶ tmp = hvx_vec_tanh_f32(vsrc[i]);              \
             vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), tmp); \
        }                                                             \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sigmoid_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sigmoid_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sigmoid_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sigmoid_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_tanh_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_tanh_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_fast_sigmoid_f16(HVX_ಸದಿಶ x_v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_one       = hvx_vec_splat_f16(1.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ em_ಮುಖವಾಡ     = Q6_Vh_vsplat_R(0x7FFF);

    // Compute absolute value of x_v
    HVX_ಸದಿಶ abs_x = Q6_V_vand_VV(x_v, em_ಮುಖವಾಡ);

    // Compute u = -abs_x * log2(e) <= 0.
    HVX_ಸದಿಶ u = hvx_vec_mul_f16_f16(abs_x, v_neg_log2e);

    // Clamp input to prevent underflow in exp2
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_clamp_min = hvx_vec_splat_f16(-24.0f);
    u = Q6_Vhf_vmax_VhfVhf(v_clamp_min, u);

    HVX_ಸದಿಶ exp_val = hvx_vec_exp2_f16(u);
    HVX_ಸದಿಶ denom   = hvx_vec_ಕೂಡಿಸು_f16_f16(v_one, exp_val);
    HVX_ಸದಿಶ sig_abs = hvx_vec_inverse_f16(denom);

    // check if x_v < 0 (using integer comparison on absolute value)
    HVX_ಸದಿಶ_Pred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);

    // If x_v < 0, return 1.0f - sig_abs
    HVX_ಸದಿಶ sig_neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sig_abs));
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(is_neg, sig_neg, sig_abs);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_tanh_f16(HVX_ಸದಿಶ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_two = hvx_vec_splat_f16(2.0f);

    HVX_ಸದಿಶ x2 = hvx_vec_mul_f16_f16(x, v_two);
    HVX_ಸದಿಶ sig2x = hvx_vec_fast_sigmoid_f16(x2);

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_neg_one = hvx_vec_splat_f16(-1.0f);
    ಹಿಂತಿರುಗಿ hvx_vec_ಕೂಡಿಸು_f16_f16(hvx_vec_mul_f16_f16(sig2x, v_two), v_neg_one);
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_sigmoid_f16_loop_body(dst_type, src_type, vec_store)         \
    ಮಾಡು {                                                                 \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                     \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                     \
                                                                         \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(_Float16);                    \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                   \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        ಗಾಗಿ (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_fast_sigmoid_f16(vsrc[i]);                \
        }                                                                \
        ಆದರೆ (nloe) {                                                      \
             HVX_ಸದಿಶ tmp = hvx_vec_fast_sigmoid_f16(vsrc[i]);         \
             vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(_Float16), tmp); \
        }                                                                \
    } ಆಗ(0)

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_tanh_f16_loop_body(dst_type, src_type, vec_store)            \
    ಮಾಡು {                                                                 \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                     \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                     \
                                                                         \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(_Float16);                    \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                   \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        ಗಾಗಿ (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_tanh_f16(vsrc[i]);                        \
        }                                                                \
        ಆದರೆ (nloe) {                                                      \
             HVX_ಸದಿಶ tmp = hvx_vec_tanh_f16(vsrc[i]);                 \
             vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(_Float16), tmp); \
        }                                                                \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sigmoid_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sigmoid_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_tanh_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_tanh_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_SIGMOID_H */
