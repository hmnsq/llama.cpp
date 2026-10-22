#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_SIGMOID_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_SIGMOID_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-inverse.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-exp.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FAST_SIGMOID_C3    (0x3f000000)  // 0.5

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_fast_sigmoid_f32(HVX_ਸਦਿਸ਼ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_ਸਦਿਸ਼ ਵਿੱਚ_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_ਸਦਿਸ਼ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(ਵਿੱਚ_int));
    HVX_ਸਦਿਸ਼ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_ਸਦਿਸ਼ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_ਸਦਿਸ਼ v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_ਸਦਿਸ਼ v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(ਵਿੱਚ_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, ਵਿੱਚ_int, 24);

    HVX_ਸਦਿਸ਼ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_ਸਦਿਸ਼ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_ਸਦਿਸ਼ res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    ਵਾਪਸ Q6_Vsf_equals_Vqf32(res);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_fast_sigmoid_f32_guard(HVX_ਸਦਿਸ਼ v,
                                                         HVX_ਸਦਿਸ਼ one,
                                                         HVX_ਸਦਿਸ਼ max_exp,
                                                         HVX_ਸਦਿਸ਼ min_exp) {
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ = hvx_vec_fast_sigmoid_f32(v);
    ਬਾਹਰਲਾ            = Q6_V_vmux_QVV(pred_max, ਬਾਹਰਲਾ, one);
    ਵਾਪਸ Q6_V_vmux_QVV(pred_min, ਬਾਹਰਲਾ, Q6_V_vzero());
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_tanh_f32(HVX_ਸਦਿਸ਼ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_ਸਦਿਸ਼ two = hvx_vec_splat_f32(2.0f);
    HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f32(1.0f);
    HVX_ਸਦਿਸ਼ x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_ਸਦਿਸ਼ max_exp = hvx_vec_splat_f32(87.f);
    HVX_ਸਦਿਸ਼ min_exp = hvx_vec_splat_f32(-87.f);

    HVX_ਸਦਿਸ਼ sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_ਸਦਿਸ਼ res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    ਵਾਪਸ Q6_Vsf_equals_Vqf32(res);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_sigmoid_loop_body(dst_type, src_type, vec_store)                                  \
    ਕਰੋ {                                                                                      \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                          \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                          \
                                                                                              \
        ਅਚਲ HVX_ਸਦਿਸ਼ one     = hvx_vec_splat_f32(1.f);                                    \
        ਅਚਲ HVX_ਸਦਿਸ਼ max_exp = hvx_vec_splat_f32(87.f);                                   \
        ਅਚਲ HVX_ਸਦਿਸ਼ min_exp = hvx_vec_splat_f32(-87.f);                                  \
                                                                                              \
        ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);                                            \
        ਅਚਲ uint32_t nvec = n / epv;                                                        \
        ਅਚਲ uint32_t nloe = n % epv;                                                        \
                                                                                              \
        uint32_t i = 0;                                                                       \
                                                                                              \
        _Pragma("unroll(4)")                                                                  \
        ਲਈ (; i < nvec; i++) {                                                               \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp);        \
        }                                                                                     \
        ਜੇ (nloe) {                                                                           \
             HVX_ਸਦਿਸ਼ tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), tmp);                         \
        }                                                                                     \
    } ਜਦੋਂ(0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_tanh_loop_body(dst_type, src_type, vec_store)             \
    ਕਰੋ {                                                              \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                  \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                  \
                                                                      \
        ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);                    \
        ਅਚਲ uint32_t nvec = n / epv;                                \
        ਅਚਲ uint32_t nloe = n % epv;                                \
                                                                      \
        uint32_t i = 0;                                               \
                                                                      \
        _Pragma("unroll(4)")                                          \
        ਲਈ (; i < nvec; i++) {                                       \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);                     \
        }                                                             \
        ਜੇ (nloe) {                                                   \
             HVX_ਸਦਿਸ਼ tmp = hvx_vec_tanh_f32(vsrc[i]);              \
             vec_store((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), tmp); \
        }                                                             \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sigmoid_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sigmoid_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sigmoid_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sigmoid_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_tanh_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_tanh_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_fast_sigmoid_f16(HVX_ਸਦਿਸ਼ x_v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one       = hvx_vec_splat_f16(1.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    ਅਚਲ HVX_ਸਦਿਸ਼ em_ਨਕਾਬ     = Q6_Vh_vsplat_R(0x7FFF);

    // Compute absolute value of x_v
    HVX_ਸਦਿਸ਼ abs_x = Q6_V_vand_VV(x_v, em_ਨਕਾਬ);

    // Compute u = -abs_x * log2(e) <= 0.
    HVX_ਸਦਿਸ਼ u = hvx_vec_mul_f16_f16(abs_x, v_neg_log2e);

    // Clamp input to prevent underflow in exp2
    ਅਚਲ HVX_ਸਦਿਸ਼ v_clamp_min = hvx_vec_splat_f16(-24.0f);
    u = Q6_Vhf_vmax_VhfVhf(v_clamp_min, u);

    HVX_ਸਦਿਸ਼ exp_val = hvx_vec_exp2_f16(u);
    HVX_ਸਦਿਸ਼ denom   = hvx_vec_ਜੋੜੋ_f16_f16(v_one, exp_val);
    HVX_ਸਦਿਸ਼ sig_abs = hvx_vec_inverse_f16(denom);

    // check if x_v < 0 (using integer comparison on absolute value)
    HVX_ਸਦਿਸ਼_Pred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);

    // If x_v < 0, return 1.0f - sig_abs
    HVX_ਸਦਿਸ਼ sig_neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sig_abs));
    ਵਾਪਸ Q6_V_vmux_QVV(is_neg, sig_neg, sig_abs);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_tanh_f16(HVX_ਸਦਿਸ਼ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    ਅਚਲ HVX_ਸਦਿਸ਼ v_two = hvx_vec_splat_f16(2.0f);

    HVX_ਸਦਿਸ਼ x2 = hvx_vec_mul_f16_f16(x, v_two);
    HVX_ਸਦਿਸ਼ sig2x = hvx_vec_fast_sigmoid_f16(x2);

    ਅਚਲ HVX_ਸਦਿਸ਼ v_neg_one = hvx_vec_splat_f16(-1.0f);
    ਵਾਪਸ hvx_vec_ਜੋੜੋ_f16_f16(hvx_vec_mul_f16_f16(sig2x, v_two), v_neg_one);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_sigmoid_f16_loop_body(dst_type, src_type, vec_store)         \
    ਕਰੋ {                                                                 \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                     \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                     \
                                                                         \
        ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(_Float16);                    \
        ਅਚਲ uint32_t nvec = n / epv;                                   \
        ਅਚਲ uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        ਲਈ (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_fast_sigmoid_f16(vsrc[i]);                \
        }                                                                \
        ਜੇ (nloe) {                                                      \
             HVX_ਸਦਿਸ਼ tmp = hvx_vec_fast_sigmoid_f16(vsrc[i]);         \
             vec_store((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(_Float16), tmp); \
        }                                                                \
    } ਜਦੋਂ(0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_tanh_f16_loop_body(dst_type, src_type, vec_store)            \
    ਕਰੋ {                                                                 \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                     \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                     \
                                                                         \
        ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(_Float16);                    \
        ਅਚਲ uint32_t nvec = n / epv;                                   \
        ਅਚਲ uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        ਲਈ (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_tanh_f16(vsrc[i]);                        \
        }                                                                \
        ਜੇ (nloe) {                                                      \
             HVX_ਸਦਿਸ਼ tmp = hvx_vec_tanh_f16(vsrc[i]);                 \
             vec_store((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(_Float16), tmp); \
        }                                                                \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sigmoid_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sigmoid_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_tanh_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_tanh_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

#ਜੇ_ਅੰਤ /* HVX_SIGMOID_H */
