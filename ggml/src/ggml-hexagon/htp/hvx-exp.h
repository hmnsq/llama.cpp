#اگر_غیر_تعریف_شدہ HVX_EXP_H
#تعریف_کرو HVX_EXP_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <math.h>

#شامل "hvx-base.h"
#شامل "hvx-floor.h"

#تعریف_کرو EXP_COEFF_5 (0x39506967)  // 0.000198757 = 1/(7!)
#تعریف_کرو EXP_COEFF_4 (0x3AB743CE)  // 0.0013982   = 1/(6!)
#تعریف_کرو EXP_COEFF_3 (0x3C088908)  // 0.00833345  = 1/(5!)
#تعریف_کرو EXP_COEFF_2 (0x3D2AA9C1)  // 0.416658    = 1/(4!)
#تعریف_کرو EXP_COEFF_1 (0x3E2AAAAA)  // 0.16666667  = 1/(3!)
#تعریف_کرو EXP_COEFF_0 (0x3F000000)  // 0.5         = 1/(2!)
#تعریف_کرو EXP_LOGN2   (0x3F317218)  // ln(2)   = 0.6931471805
#تعریف_کرو EXP_LOG2E   (0x3FB8AA3B)  // log2(e) = 1/ln(2) = 1.4426950408
#تعریف_کرو EXP_LOG2E_F 1.44269504f
#تعریف_کرو EXP_ONE     (0x3f800000)  // 1.0
#تعریف_کرو EXP_RANGE_R (0x42B17218)  // ln(FLT_MAX) approx = 88.7228
#تعریف_کرو EXP_RANGE_L (0xC2B00000)  // -88.0 (approx log(FLT_MIN))

ساکن اندرونی HVX_سمتیہ hvx_vec_exp_f32(HVX_سمتیہ اندر_vec) {
    HVX_سمتیہ z_qf32_v;
    HVX_سمتیہ x_v;
    HVX_سمتیہ x_qf32_v;
    HVX_سمتیہ y_v;
    HVX_سمتیہ k_v;
    HVX_سمتیہ f_v;
    HVX_سمتیہ epsilon_v;
    HVX_سمتیہ log2e = Q6_V_vsplat_R(EXP_LOG2E);
    HVX_سمتیہ logn2 = Q6_V_vsplat_R(EXP_LOGN2);
    HVX_سمتیہ E_const;
    HVX_سمتیہ zero_v = Q6_V_vzero();

    // exp(x) is approximated as follows:
    //   f = floor(x/ln(2)) = floor(x*log2(e))
    //   epsilon = x - f*ln(2)
    //   exp(x) = exp(epsilon+f*ln(2))
    //          = exp(epsilon)*exp(f*ln(2))
    //          = exp(epsilon)*2^f
    //
    //   Since epsilon is close to zero, it can be approximated with its Taylor series:
    //            exp(x) ~= 1+x+x^2/2!+x^3/3!+...+x^n/n!+...
    //   Preserving the first eight elements, we get:
    //            exp(x) ~= 1+x+e0*x^2+e1*x^3+e2*x^4+e3*x^5+e4*x^6+e5*x^7
    //                   =  1+x+(E0+(E1+(E2+(E3+(E4+E5*x)*x)*x)*x)*x)*x^2

    HVX_سمتیہ عارضی_v = اندر_vec;

    // Clamp inputs to (-88.0, 88.0) to avoid overflow/underflow
    HVX_سمتیہ_Pred pred_cap_دایاں = Q6_Q_vcmp_gt_VsfVsf(اندر_vec, Q6_V_vsplat_R(EXP_RANGE_R));
    HVX_سمتیہ_Pred pred_cap_بایاں  = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vsplat_R(EXP_RANGE_L), اندر_vec);

    اندر_vec = Q6_V_vmux_QVV(pred_cap_دایاں, Q6_V_vsplat_R(EXP_RANGE_R), عارضی_v);
    اندر_vec = Q6_V_vmux_QVV(pred_cap_بایاں, Q6_V_vsplat_R(EXP_RANGE_L), اندر_vec);

    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, اندر_vec);
    epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = hvx_vec_floor_f32(epsilon_v);
    k_v = hvx_vec_truncate_f32(f_v);

    x_qf32_v = Q6_Vqf32_vadd_VsfVsf(اندر_vec, zero_v);

    //  x = x - f_v * logn2;
    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
    x_qf32_v  = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
    // normalize before every QFloat's vmpy
    x_qf32_v  = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);

    x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

    // z = x * x;
    z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
    z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

    // y = E4 + E5 * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_5);
    y_v     = Q6_Vqf32_vmpy_VsfVsf(E_const, x_v);
    E_const = Q6_V_vsplat_R(EXP_COEFF_4);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E3 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_3);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E2 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_2);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E1 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_1);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E0 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_0);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = x + y * z;
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = y + 1.0;
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(EXP_ONE));

    // insert exponents
    //        y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent

    y_v = Q6_Vsf_equals_Vqf32(y_v);

    // add k_v to the exponent of y_v
    HVX_سمتیہ y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);

    y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, IEEE_VSF_MANTLEN + 1);
    y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

    // exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_سمتیہ_Pred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

    y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, IEEE_VSF_MANTLEN);

    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);

    واپس y_v;
}

ساکن اندرونی HVX_سمتیہ hvx_vec_exp_f32_guard(HVX_سمتیہ اندر_vec, HVX_سمتیہ max_exp, HVX_سمتیہ inf) {
    ثابت HVX_سمتیہ_Pred pred0 = Q6_Q_vcmp_gt_VsfVsf(اندر_vec, max_exp);

    HVX_سمتیہ برآمد = hvx_vec_exp_f32(اندر_vec);

    واپس Q6_V_vmux_QVV(pred0, inf, برآمد);
}

ساکن اندرونی خالی hvx_exp_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح عدد_elems, bool negate) {
    صحیح بایاں_over       = عدد_elems & (VLEN_FP32 - 1);
    صحیح عدد_elems_پورا = عدد_elems - بایاں_over;

    صحیح unaligned_addr = 0;
    صحیح unaligned_loop = 0;
    اگر ((0 == hex_is_aligned((خالی *) src, VLEN)) || (0 == hex_is_aligned((خالی *) dst, VLEN))) {
        unaligned_addr = 1;
    }
    // assert((0 == unaligned_addr) || (0 == num_elems_whole));
    اگر ((1 == unaligned_addr) && (عدد_elems_پورا != 0)) {
        unaligned_loop = 1;
    }

    HVX_سمتیہ vec_برآمد = Q6_V_vzero();

    ساکن ثابت اعشاری kInf    = INFINITY;
    ساکن ثابت اعشاری kMaxExp = 88.7228f;

    ثابت HVX_سمتیہ max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_سمتیہ inf     = hvx_vec_splat_f32(kInf);

    اگر (0 == unaligned_loop) {
        HVX_سمتیہ * p_vec_in1 = (HVX_سمتیہ *) src;
        HVX_سمتیہ * p_vec_برآمد = (HVX_سمتیہ *) dst;

        #ہدایت unroll(2)
        برائے (صحیح i = 0; i < عدد_elems_پورا; i += VLEN_FP32) {
            اگر (true == negate) {
                HVX_سمتیہ neg_vec_اندر = hvx_vec_neg_f32(*p_vec_in1++);
                *p_vec_برآمد++          = hvx_vec_exp_f32_guard(neg_vec_اندر, max_exp, inf);
            } ورنہ {
                *p_vec_برآمد++ = hvx_vec_exp_f32_guard(*p_vec_in1++, max_exp, inf);
            }
        }
    } ورنہ {
        #ہدایت unroll(2)
        برائے (صحیح i = 0; i < عدد_elems_پورا; i += VLEN_FP32) {
            HVX_سمتیہ اندر = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            اگر (true == negate) {
                HVX_سمتیہ neg_vec_اندر                    = hvx_vec_neg_f32(اندر);
                *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_exp_f32_guard(neg_vec_اندر, max_exp, inf);
            } ورنہ {
                *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_exp_f32_guard(اندر, max_exp, inf);
            }
        }
    }

    اگر (بایاں_over > 0) {
        ثابت اعشاری * srcf = (اعشاری *) src + عدد_elems_پورا;
        اعشاری *       dstf = (اعشاری *) dst + عدد_elems_پورا;

        HVX_سمتیہ اندر = *(HVX_UVector *) srcf;

        اگر (true == negate) {
            HVX_سمتیہ neg_vec_اندر = hvx_vec_neg_f32(اندر);

            vec_برآمد = hvx_vec_exp_f32_guard(neg_vec_اندر, max_exp, inf);
        } ورنہ {
            vec_برآمد = hvx_vec_exp_f32_guard(اندر, max_exp, inf);
        }

        hvx_vec_store_u((خالی *) dstf, بایاں_over * SIZEOF_FP32, vec_برآمد);
    }
}

ساکن اندرونی HVX_سمتیہ hvx_vec_exp2_f16(HVX_سمتیہ x_v) {
    ثابت HVX_سمتیہ zero_v    = Q6_V_vzero();
    ثابت HVX_سمتیہ half_hf_v = Q6_Vh_vsplat_R(0x3800);  // fp16 0.5

    // Clamp input to prevent integer underflow in FP16-to-INT16 conversion
    ثابت HVX_سمتیہ v_clamp_min = hvx_vec_splat_f16(-24.0f);
    x_v = Q6_Vhf_vmax_VhfVhf(v_clamp_min, x_v);

    // k = round_toward_neg_inf(x);  f = (float)k;  frac = x - f
    HVX_سمتیہ x_minus_half = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x_v, half_hf_v));
    HVX_سمتیہ k_v          = Q6_Vh_equals_Vhf(x_minus_half);  // truncate to int16
    HVX_سمتیہ f_v          = Q6_Vhf_equals_Vh(k_v);           // back to fp16

    HVX_سمتیہ x_qf16 = Q6_Vqf16_vsub_VhfVhf(x_v, f_v);        // fractional part in qf16

    // Horner: y = ((((E5*x + E4)*x + E3)*x + E2)*x + E1)*x + E0
    HVX_سمتیہ y = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vh_vsplat_R(0x5082), x_qf16); // E5*x
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x157d));        // + E4
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x20ed));        // + E3
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x2b1b));        // + E2
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x33b0));        // + E1
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x398c));        // + E0
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);                      // y = y * x
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x3c00));        // + 1.0

    // Combine polynomial (mantissa) with integer part (exponent): result = y * 2^k
    y                          = Q6_Vhf_equals_Vqf16(y);
    HVX_سمتیہ y_exp           = Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(y, 1), 11);
    y_exp                      = Q6_Vh_vadd_VhVh(k_v, y_exp);
    HVX_سمتیہ_Pred q_underflow = Q6_Q_vcmp_gt_VhVh(zero_v, y_exp);
    y                          = Q6_Vh_vaslacc_VhVhR(y, k_v, 10);
    واپس Q6_V_vmux_QVV(q_underflow, zero_v, y);
}

#اگر_ختم /* HVX_EXP_H */
