#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_FLOOR_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_FLOOR_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IEEE_VSF_EXPLEN   (8)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IEEE_VSF_EXPBIAS  (127)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IEEE_VSF_EXPMASK  (0xFF)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IEEE_VSF_MANTLEN  (23)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IEEE_VSF_MANTMASK (0x7FFFFF)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IEEE_VSF_MIMPMASK (0x800000)

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_truncate_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ_vec) {
    HVX_ਸਦਿਸ਼ ਨਕਾਬ_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_ਸਦਿਸ਼ ਨਕਾਬ_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_ਸਦਿਸ਼ const_zero_v = Q6_V_vzero();

    HVX_ਸਦਿਸ਼_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, ਵਿੱਚ_vec);

    HVX_ਸਦਿਸ਼ expval_v = ਵਿੱਚ_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_ਸਦਿਸ਼_Pred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_ਸਦਿਸ਼ rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_ਸਦਿਸ਼ mant_v = ਵਿੱਚ_vec & ਨਕਾਬ_mant_v;                  // obtain mantissa
    HVX_ਸਦਿਸ਼ vout   = Q6_Vw_vadd_VwVw(mant_v, ਨਕਾਬ_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_ਸਦਿਸ਼ neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    ਵਾਪਸ (vout);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_floor_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ_vec) {
    HVX_ਸਦਿਸ਼ ਨਕਾਬ_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_ਸਦਿਸ਼ ਨਕਾਬ_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_ਸਦਿਸ਼ const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_ਸਦਿਸ਼ const_zero_v   = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_ਸਦਿਸ਼_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, ਵਿੱਚ_vec);

    HVX_ਸਦਿਸ਼ expval_v = ਵਿੱਚ_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_ਸਦਿਸ਼_Pred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_ਸਦਿਸ਼_Pred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_ਸਦਿਸ਼_Pred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, ਵਿੱਚ_vec, const_zero_v);
    HVX_ਸਦਿਸ਼_Pred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, ਵਿੱਚ_vec);

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
    ਨਕਾਬ_mant_v >>= expval_v;
    HVX_ਸਦਿਸ਼ neg_addin_v    = ਨਕਾਬ_impl_v >> expval_v;
    HVX_ਸਦਿਸ਼ vout_neg_addin = Q6_Vw_vadd_VwVw(ਵਿੱਚ_vec, neg_addin_v);
    HVX_ਸਦਿਸ਼ vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, ਵਿੱਚ_vec);

    HVX_ਸਦਿਸ਼     ਨਕਾਬ_chk_v = Q6_V_vand_VV(ਵਿੱਚ_vec, ਨਕਾਬ_mant_v);  // chk if bits set
    HVX_ਸਦਿਸ਼_Pred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, ਨਕਾਬ_chk_v);

    HVX_ਸਦਿਸ਼ not_ਨਕਾਬ_v = Q6_V_vnot_V(ਨਕਾਬ_mant_v);        // frac bits to clear
    HVX_ਸਦਿਸ਼ vfrfloor_v = Q6_V_vand_VV(vout, not_ਨਕਾਬ_v);  // clear frac bits

    vout = ਵਿੱਚ_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, ਵਿੱਚ_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    ਵਾਪਸ vout;
}

#ਜੇ_ਅੰਤ /* HVX_FLOOR_H */
