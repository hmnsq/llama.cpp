#ifndef HVX_SIGMOID_H
#define HVX_SIGMOID_H

#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hvx-inverse.h"
#अन्तर्भाव "hvx-exp.h"

#define FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#define FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#define FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#define FAST_SIGMOID_C3    (0x3f000000)  // 0.5

स्थिर अन्तःस्थ HVX_Vector hvx_vec_fast_sigmoid_f32(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    पटिनिवत्त Q6_Vsf_equals_Vqf32(res);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_fast_sigmoid_f32_guard(HVX_Vector v,
                                                         HVX_Vector one,
                                                         HVX_Vector max_exp,
                                                         HVX_Vector min_exp) {
    ध्रुव HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ध्रुव HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    पटिनिवत्त Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_tanh_f32(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_Vector two = hvx_vec_splat_f32(2.0f);
    HVX_Vector one = hvx_vec_splat_f32(1.0f);
    HVX_Vector x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);

    HVX_Vector sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    पटिनिवत्त Q6_Vsf_equals_Vqf32(res);
}

#define hvx_sigmoid_loop_body(dst_type, src_type, vec_store)                                  \
    करोहि {                                                                                      \
        dst_type * restrict vdst = (dst_type *) dst;                                          \
        src_type * restrict vsrc = (src_type *) src;                                          \
                                                                                              \
        ध्रुव HVX_Vector one     = hvx_vec_splat_f32(1.f);                                    \
        ध्रुव HVX_Vector max_exp = hvx_vec_splat_f32(87.f);                                   \
        ध्रुव HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);                                  \
                                                                                              \
        ध्रुव uint32_t epv  = 128 / परिमाण(भिन्न);                                            \
        ध्रुव uint32_t nvec = n / epv;                                                        \
        ध्रुव uint32_t nloe = n % epv;                                                        \
                                                                                              \
        uint32_t i = 0;                                                                       \
                                                                                              \
        _Pragma("unroll(4)")                                                                  \
        अत्थाय (; i < nvec; i++) {                                                               \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp);        \
        }                                                                                     \
        सचे (nloe) {                                                                           \
             HVX_Vector tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((शून्य *) &vdst[i], nloe * परिमाण(भिन्न), tmp);                         \
        }                                                                                     \
    } याव(0)

#define hvx_tanh_loop_body(dst_type, src_type, vec_store)             \
    करोहि {                                                              \
        dst_type * restrict vdst = (dst_type *) dst;                  \
        src_type * restrict vsrc = (src_type *) src;                  \
                                                                      \
        ध्रुव uint32_t epv  = 128 / परिमाण(भिन्न);                    \
        ध्रुव uint32_t nvec = n / epv;                                \
        ध्रुव uint32_t nloe = n % epv;                                \
                                                                      \
        uint32_t i = 0;                                               \
                                                                      \
        _Pragma("unroll(4)")                                          \
        अत्थाय (; i < nvec; i++) {                                       \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);                     \
        }                                                             \
        सचे (nloe) {                                                   \
             HVX_Vector tmp = hvx_vec_tanh_f32(vsrc[i]);              \
             vec_store((शून्य *) &vdst[i], nloe * परिमाण(भिन्न), tmp); \
        }                                                             \
    } याव(0)

स्थिर अन्तःस्थ शून्य hvx_sigmoid_f32_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sigmoid_f32_au(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sigmoid_f32_ua(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sigmoid_f32_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_tanh_f32_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_tanh_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_fast_sigmoid_f16(HVX_Vector x_v) {
    ध्रुव HVX_Vector v_one       = hvx_vec_splat_f16(1.0f);
    ध्रुव HVX_Vector v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    ध्रुव HVX_Vector em_mask     = Q6_Vh_vsplat_R(0x7FFF);

    // Compute absolute value of x_v
    HVX_Vector abs_x = Q6_V_vand_VV(x_v, em_mask);

    // Compute u = -abs_x * log2(e) <= 0.
    HVX_Vector u = hvx_vec_mul_f16_f16(abs_x, v_neg_log2e);

    // Clamp input to prevent underflow in exp2
    ध्रुव HVX_Vector v_clamp_min = hvx_vec_splat_f16(-24.0f);
    u = Q6_Vhf_vmax_VhfVhf(v_clamp_min, u);

    HVX_Vector exp_val = hvx_vec_exp2_f16(u);
    HVX_Vector denom   = hvx_vec_add_f16_f16(v_one, exp_val);
    HVX_Vector sig_abs = hvx_vec_inverse_f16(denom);

    // check if x_v < 0 (using integer comparison on absolute value)
    HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);

    // If x_v < 0, return 1.0f - sig_abs
    HVX_Vector sig_neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sig_abs));
    पटिनिवत्त Q6_V_vmux_QVV(is_neg, sig_neg, sig_abs);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_tanh_f16(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    ध्रुव HVX_Vector v_two = hvx_vec_splat_f16(2.0f);

    HVX_Vector x2 = hvx_vec_mul_f16_f16(x, v_two);
    HVX_Vector sig2x = hvx_vec_fast_sigmoid_f16(x2);

    ध्रुव HVX_Vector v_neg_one = hvx_vec_splat_f16(-1.0f);
    पटिनिवत्त hvx_vec_add_f16_f16(hvx_vec_mul_f16_f16(sig2x, v_two), v_neg_one);
}

#define hvx_sigmoid_f16_loop_body(dst_type, src_type, vec_store)         \
    करोहि {                                                                 \
        dst_type * restrict vdst = (dst_type *) dst;                     \
        src_type * restrict vsrc = (src_type *) src;                     \
                                                                         \
        ध्रुव uint32_t epv  = 128 / परिमाण(_Float16);                    \
        ध्रुव uint32_t nvec = n / epv;                                   \
        ध्रुव uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        अत्थाय (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_fast_sigmoid_f16(vsrc[i]);                \
        }                                                                \
        सचे (nloe) {                                                      \
             HVX_Vector tmp = hvx_vec_fast_sigmoid_f16(vsrc[i]);         \
             vec_store((शून्य *) &vdst[i], nloe * परिमाण(_Float16), tmp); \
        }                                                                \
    } याव(0)

#define hvx_tanh_f16_loop_body(dst_type, src_type, vec_store)            \
    करोहि {                                                                 \
        dst_type * restrict vdst = (dst_type *) dst;                     \
        src_type * restrict vsrc = (src_type *) src;                     \
                                                                         \
        ध्रुव uint32_t epv  = 128 / परिमाण(_Float16);                    \
        ध्रुव uint32_t nvec = n / epv;                                   \
        ध्रुव uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        अत्थाय (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_tanh_f16(vsrc[i]);                        \
        }                                                                \
        सचे (nloe) {                                                      \
             HVX_Vector tmp = hvx_vec_tanh_f16(vsrc[i]);                 \
             vec_store((शून्य *) &vdst[i], nloe * परिमाण(_Float16), tmp); \
        }                                                                \
    } याव(0)

स्थिर अन्तःस्थ शून्य hvx_sigmoid_f16_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_sigmoid_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_tanh_f16_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_tanh_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

#endif /* HVX_SIGMOID_H */
