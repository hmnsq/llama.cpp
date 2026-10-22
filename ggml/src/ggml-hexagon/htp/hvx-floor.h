#اگر_تعریف_نشده HVX_FLOOR_H
#تعریف_کن HVX_FLOOR_H

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن "hvx-base.h"

#تعریف_کن IEEE_VSF_EXPLEN   (8)
#تعریف_کن IEEE_VSF_EXPBIAS  (127)
#تعریف_کن IEEE_VSF_EXPMASK  (0xFF)
#تعریف_کن IEEE_VSF_MANTLEN  (23)
#تعریف_کن IEEE_VSF_MANTMASK (0x7FFFFF)
#تعریف_کن IEEE_VSF_MIMPMASK (0x800000)

ساكن مضمن HVX_وکتور hvx_vec_truncate_f32(HVX_وکتور درون_vec) {
    HVX_وکتور ماسک_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_وکتور ماسک_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_وکتور const_zero_v = Q6_V_vzero();

    HVX_وکتور_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, درون_vec);

    HVX_وکتور expval_v = درون_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_وکتور_Pred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_وکتور rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_وکتور mant_v = درون_vec & ماسک_mant_v;                  // obtain mantissa
    HVX_وکتور vout   = Q6_Vw_vadd_VwVw(mant_v, ماسک_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_وکتور neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    أرجع (vout);
}

ساكن مضمن HVX_وکتور hvx_vec_floor_f32(HVX_وکتور درون_vec) {
    HVX_وکتور ماسک_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_وکتور ماسک_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_وکتور const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_وکتور const_zero_v   = Q6_V_vzero();
    HVX_وکتور const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_وکتور_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, درون_vec);

    HVX_وکتور expval_v = درون_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_وکتور_Pred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_وکتور_Pred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_وکتور_Pred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, درون_vec, const_zero_v);
    HVX_وکتور_Pred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, درون_vec);

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
    ماسک_mant_v >>= expval_v;
    HVX_وکتور neg_addin_v    = ماسک_impl_v >> expval_v;
    HVX_وکتور vout_neg_addin = Q6_Vw_vadd_VwVw(درون_vec, neg_addin_v);
    HVX_وکتور vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, درون_vec);

    HVX_وکتور     ماسک_chk_v = Q6_V_vand_VV(درون_vec, ماسک_mant_v);  // chk if bits set
    HVX_وکتور_Pred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, ماسک_chk_v);

    HVX_وکتور not_ماسک_v = Q6_V_vnot_V(ماسک_mant_v);        // frac bits to clear
    HVX_وکتور vfrfloor_v = Q6_V_vand_VV(vout, not_ماسک_v);  // clear frac bits

    vout = درون_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, درون_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    أرجع vout;
}

#پایان_اگر /* HVX_FLOOR_H */
