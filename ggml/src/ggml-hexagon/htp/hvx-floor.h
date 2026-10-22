#यदि_अपरिभाषितम् HVX_FLOOR_H
#परिभाषय HVX_FLOOR_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव "hvx-base.h"

#परिभाषय IEEE_VSF_EXPLEN   (8)
#परिभाषय IEEE_VSF_EXPBIAS  (127)
#परिभाषय IEEE_VSF_EXPMASK  (0xFF)
#परिभाषय IEEE_VSF_MANTLEN  (23)
#परिभाषय IEEE_VSF_MANTMASK (0x7FFFFF)
#परिभाषय IEEE_VSF_MIMPMASK (0x800000)

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_truncate_f32(HVX_सदिशम् अन्तर्_vec) {
    HVX_सदिशम् आवरणम्_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_सदिशम् आवरणम्_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_सदिशम् const_zero_v = Q6_V_vzero();

    HVX_सदिशम्_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, अन्तर्_vec);

    HVX_सदिशम् expval_v = अन्तर्_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_सदिशम्_Pred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_सदिशम् rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_सदिशम् mant_v = अन्तर्_vec & आवरणम्_mant_v;                  // obtain mantissa
    HVX_सदिशम् vout   = Q6_Vw_vadd_VwVw(mant_v, आवरणम्_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_सदिशम् neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    प्रत्यागम (vout);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_floor_f32(HVX_सदिशम् अन्तर्_vec) {
    HVX_सदिशम् आवरणम्_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_सदिशम् आवरणम्_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_सदिशम् const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_सदिशम् const_zero_v   = Q6_V_vzero();
    HVX_सदिशम् const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_सदिशम्_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, अन्तर्_vec);

    HVX_सदिशम् expval_v = अन्तर्_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_सदिशम्_Pred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_सदिशम्_Pred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_सदिशम्_Pred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, अन्तर्_vec, const_zero_v);
    HVX_सदिशम्_Pred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, अन्तर्_vec);

    // if expval < 0 (q_negexp)         // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval                // fraction bits to mask off
    //    vout = ~(mask)                // apply mask to remove fraction
    //    if (qneg)                     // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                             // already an integer
    //    ;                             // no change

    // compute floor
    आवरणम्_mant_v >>= expval_v;
    HVX_सदिशम् neg_addin_v    = आवरणम्_impl_v >> expval_v;
    HVX_सदिशम् vout_neg_addin = Q6_Vw_vadd_VwVw(अन्तर्_vec, neg_addin_v);
    HVX_सदिशम् vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, अन्तर्_vec);

    HVX_सदिशम्     आवरणम्_chk_v = Q6_V_vand_VV(अन्तर्_vec, आवरणम्_mant_v);  // chk if bits set
    HVX_सदिशम्_Pred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, आवरणम्_chk_v);

    HVX_सदिशम् not_आवरणम्_v = Q6_V_vnot_V(आवरणम्_mant_v);        // frac bits to clear
    HVX_सदिशम् vfrfloor_v = Q6_V_vand_VV(vout, not_आवरणम्_v);  // clear frac bits

    vout = अन्तर्_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, अन्तर्_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    प्रत्यागम vout;
}

#यदि_अन्तः /* HVX_FLOOR_H */
