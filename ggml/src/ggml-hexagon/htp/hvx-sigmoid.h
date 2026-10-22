#اگر_غیر_تعریف_شدہ HVX_SIGMOID_H
#تعریف_کرو HVX_SIGMOID_H

#شامل "hvx-base.h"
#شامل "hvx-inverse.h"
#شامل "hvx-exp.h"

#تعریف_کرو FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#تعریف_کرو FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#تعریف_کرو FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#تعریف_کرو FAST_SIGMOID_C3    (0x3f000000)  // 0.5

ساکن اندرونی HVX_سمتیہ hvx_vec_fast_sigmoid_f32(HVX_سمتیہ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_سمتیہ اندر_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_سمتیہ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(اندر_int));
    HVX_سمتیہ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_سمتیہ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_سمتیہ v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_سمتیہ v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(اندر_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, اندر_int, 24);

    HVX_سمتیہ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_سمتیہ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_سمتیہ res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    واپس Q6_Vsf_equals_Vqf32(res);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_fast_sigmoid_f32_guard(HVX_سمتیہ v,
                                                         HVX_سمتیہ one,
                                                         HVX_سمتیہ max_exp,
                                                         HVX_سمتیہ min_exp) {
    ثابت HVX_سمتیہ_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ثابت HVX_سمتیہ_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_سمتیہ برآمد = hvx_vec_fast_sigmoid_f32(v);
    برآمد            = Q6_V_vmux_QVV(pred_max, برآمد, one);
    واپس Q6_V_vmux_QVV(pred_min, برآمد, Q6_V_vzero());
}

ساکن اندرونی HVX_سمتیہ hvx_vec_tanh_f32(HVX_سمتیہ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_سمتیہ two = hvx_vec_splat_f32(2.0f);
    HVX_سمتیہ one = hvx_vec_splat_f32(1.0f);
    HVX_سمتیہ x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_سمتیہ max_exp = hvx_vec_splat_f32(87.f);
    HVX_سمتیہ min_exp = hvx_vec_splat_f32(-87.f);

    HVX_سمتیہ sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_سمتیہ res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    واپس Q6_Vsf_equals_Vqf32(res);
}

#تعریف_کرو hvx_sigmoid_loop_body(dst_type, src_type, vec_store)                                  \
    کرو {                                                                                      \
        dst_type * محدود vdst = (dst_type *) dst;                                          \
        src_type * محدود vsrc = (src_type *) src;                                          \
                                                                                              \
        ثابت HVX_سمتیہ one     = hvx_vec_splat_f32(1.f);                                    \
        ثابت HVX_سمتیہ max_exp = hvx_vec_splat_f32(87.f);                                   \
        ثابت HVX_سمتیہ min_exp = hvx_vec_splat_f32(-87.f);                                  \
                                                                                              \
        ثابت uint32_t epv  = 128 / حجم(اعشاری);                                            \
        ثابت uint32_t nvec = n / epv;                                                        \
        ثابت uint32_t nloe = n % epv;                                                        \
                                                                                              \
        uint32_t i = 0;                                                                       \
                                                                                              \
        _Pragma("unroll(4)")                                                                  \
        برائے (; i < nvec; i++) {                                                               \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp);        \
        }                                                                                     \
        اگر (nloe) {                                                                           \
             HVX_سمتیہ tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((خالی *) &vdst[i], nloe * حجم(اعشاری), tmp);                         \
        }                                                                                     \
    } جب(0)

#تعریف_کرو hvx_tanh_loop_body(dst_type, src_type, vec_store)             \
    کرو {                                                              \
        dst_type * محدود vdst = (dst_type *) dst;                  \
        src_type * محدود vsrc = (src_type *) src;                  \
                                                                      \
        ثابت uint32_t epv  = 128 / حجم(اعشاری);                    \
        ثابت uint32_t nvec = n / epv;                                \
        ثابت uint32_t nloe = n % epv;                                \
                                                                      \
        uint32_t i = 0;                                               \
                                                                      \
        _Pragma("unroll(4)")                                          \
        برائے (; i < nvec; i++) {                                       \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);                     \
        }                                                             \
        اگر (nloe) {                                                   \
             HVX_سمتیہ tmp = hvx_vec_tanh_f32(vsrc[i]);              \
             vec_store((خالی *) &vdst[i], nloe * حجم(اعشاری), tmp); \
        }                                                             \
    } جب(0)

ساکن اندرونی خالی hvx_sigmoid_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_sigmoid_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_sigmoid_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_sigmoid_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_tanh_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_tanh_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_fast_sigmoid_f16(HVX_سمتیہ x_v) {
    ثابت HVX_سمتیہ v_one       = hvx_vec_splat_f16(1.0f);
    ثابت HVX_سمتیہ v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    ثابت HVX_سمتیہ em_نقاب     = Q6_Vh_vsplat_R(0x7FFF);

    // Compute absolute value of x_v
    HVX_سمتیہ abs_x = Q6_V_vand_VV(x_v, em_نقاب);

    // Compute u = -abs_x * log2(e) <= 0.
    HVX_سمتیہ u = hvx_vec_mul_f16_f16(abs_x, v_neg_log2e);

    // Clamp input to prevent underflow in exp2
    ثابت HVX_سمتیہ v_clamp_min = hvx_vec_splat_f16(-24.0f);
    u = Q6_Vhf_vmax_VhfVhf(v_clamp_min, u);

    HVX_سمتیہ exp_val = hvx_vec_exp2_f16(u);
    HVX_سمتیہ denom   = hvx_vec_جوڑو_f16_f16(v_one, exp_val);
    HVX_سمتیہ sig_abs = hvx_vec_inverse_f16(denom);

    // check if x_v < 0 (using integer comparison on absolute value)
    HVX_سمتیہ_Pred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);

    // If x_v < 0, return 1.0f - sig_abs
    HVX_سمتیہ sig_neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sig_abs));
    واپس Q6_V_vmux_QVV(is_neg, sig_neg, sig_abs);
}

ساکن اندرونی HVX_سمتیہ hvx_vec_tanh_f16(HVX_سمتیہ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    ثابت HVX_سمتیہ v_two = hvx_vec_splat_f16(2.0f);

    HVX_سمتیہ x2 = hvx_vec_mul_f16_f16(x, v_two);
    HVX_سمتیہ sig2x = hvx_vec_fast_sigmoid_f16(x2);

    ثابت HVX_سمتیہ v_neg_one = hvx_vec_splat_f16(-1.0f);
    واپس hvx_vec_جوڑو_f16_f16(hvx_vec_mul_f16_f16(sig2x, v_two), v_neg_one);
}

#تعریف_کرو hvx_sigmoid_f16_loop_body(dst_type, src_type, vec_store)         \
    کرو {                                                                 \
        dst_type * محدود vdst = (dst_type *) dst;                     \
        src_type * محدود vsrc = (src_type *) src;                     \
                                                                         \
        ثابت uint32_t epv  = 128 / حجم(_Float16);                    \
        ثابت uint32_t nvec = n / epv;                                   \
        ثابت uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        برائے (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_fast_sigmoid_f16(vsrc[i]);                \
        }                                                                \
        اگر (nloe) {                                                      \
             HVX_سمتیہ tmp = hvx_vec_fast_sigmoid_f16(vsrc[i]);         \
             vec_store((خالی *) &vdst[i], nloe * حجم(_Float16), tmp); \
        }                                                                \
    } جب(0)

#تعریف_کرو hvx_tanh_f16_loop_body(dst_type, src_type, vec_store)            \
    کرو {                                                                 \
        dst_type * محدود vdst = (dst_type *) dst;                     \
        src_type * محدود vsrc = (src_type *) src;                     \
                                                                         \
        ثابت uint32_t epv  = 128 / حجم(_Float16);                    \
        ثابت uint32_t nvec = n / epv;                                   \
        ثابت uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        برائے (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_tanh_f16(vsrc[i]);                        \
        }                                                                \
        اگر (nloe) {                                                      \
             HVX_سمتیہ tmp = hvx_vec_tanh_f16(vsrc[i]);                 \
             vec_store((خالی *) &vdst[i], nloe * حجم(_Float16), tmp); \
        }                                                                \
    } جب(0)

ساکن اندرونی خالی hvx_sigmoid_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_sigmoid_f16_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_tanh_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_tanh_f16_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

#اگر_ختم /* HVX_SIGMOID_H */
