#جيڪڏهن_وصف_نه_ٿيل HVX_FLOOR_H
#وصف_ڪريو HVX_FLOOR_H

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن "hvx-base.h"

#وصف_ڪريو IEEE_VSF_EXPLEN   (8)
#وصف_ڪريو IEEE_VSF_EXPBIAS  (127)
#وصف_ڪريو IEEE_VSF_EXPMASK  (0xFF)
#وصف_ڪريو IEEE_VSF_MANTLEN  (23)
#وصف_ڪريو IEEE_VSF_MANTMASK (0x7FFFFF)
#وصف_ڪريو IEEE_VSF_MIMPMASK (0x800000)

ساكن مضمن HVX_ويڪٽر hvx_vec_truncate_f32(HVX_ويڪٽر اندريون_vec) {
    HVX_ويڪٽر نقاب_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_ويڪٽر نقاب_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_ويڪٽر const_zero_v = Q6_V_vzero();

    HVX_ويڪٽر_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, اندريون_vec);

    HVX_ويڪٽر expval_v = اندريون_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_ويڪٽر_Pred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_ويڪٽر rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_ويڪٽر mant_v = اندريون_vec & نقاب_mant_v;                  // obtain mantissa
    HVX_ويڪٽر vout   = Q6_Vw_vadd_VwVw(mant_v, نقاب_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_ويڪٽر neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    أرجع (vout);
}

ساكن مضمن HVX_ويڪٽر hvx_vec_floor_f32(HVX_ويڪٽر اندريون_vec) {
    HVX_ويڪٽر نقاب_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_ويڪٽر نقاب_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_ويڪٽر const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_ويڪٽر const_zero_v   = Q6_V_vzero();
    HVX_ويڪٽر const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_ويڪٽر_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, اندريون_vec);

    HVX_ويڪٽر expval_v = اندريون_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_ويڪٽر_Pred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_ويڪٽر_Pred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_ويڪٽر_Pred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, اندريون_vec, const_zero_v);
    HVX_ويڪٽر_Pred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, اندريون_vec);

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
    نقاب_mant_v >>= expval_v;
    HVX_ويڪٽر neg_addin_v    = نقاب_impl_v >> expval_v;
    HVX_ويڪٽر vout_neg_addin = Q6_Vw_vadd_VwVw(اندريون_vec, neg_addin_v);
    HVX_ويڪٽر vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, اندريون_vec);

    HVX_ويڪٽر     نقاب_chk_v = Q6_V_vand_VV(اندريون_vec, نقاب_mant_v);  // chk if bits set
    HVX_ويڪٽر_Pred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, نقاب_chk_v);

    HVX_ويڪٽر not_نقاب_v = Q6_V_vnot_V(نقاب_mant_v);        // frac bits to clear
    HVX_ويڪٽر vfrfloor_v = Q6_V_vand_VV(vout, not_نقاب_v);  // clear frac bits

    vout = اندريون_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, اندريون_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    أرجع vout;
}

#جيڪڏهن_پورو /* HVX_FLOOR_H */
