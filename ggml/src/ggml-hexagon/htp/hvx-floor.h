#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_FLOOR_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_FLOOR_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hvx-base.h"

#ವ್ಯಾಖ್ಯಾನಿಸು IEEE_VSF_EXPLEN   (8)
#ವ್ಯಾಖ್ಯಾನಿಸು IEEE_VSF_EXPBIAS  (127)
#ವ್ಯಾಖ್ಯಾನಿಸು IEEE_VSF_EXPMASK  (0xFF)
#ವ್ಯಾಖ್ಯಾನಿಸು IEEE_VSF_MANTLEN  (23)
#ವ್ಯಾಖ್ಯಾನಿಸು IEEE_VSF_MANTMASK (0x7FFFFF)
#ವ್ಯಾಖ್ಯಾನಿಸು IEEE_VSF_MIMPMASK (0x800000)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_truncate_f32(HVX_ಸದಿಶ ಒಳ_vec) {
    HVX_ಸದಿಶ ಮುಖವಾಡ_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_ಸದಿಶ ಮುಖವಾಡ_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_ಸದಿಶ const_zero_v = Q6_V_vzero();

    HVX_ಸದಿಶ_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, ಒಳ_vec);

    HVX_ಸದಿಶ expval_v = ಒಳ_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_ಸದಿಶ_Pred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_ಸದಿಶ rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_ಸದಿಶ mant_v = ಒಳ_vec & ಮುಖವಾಡ_mant_v;                  // obtain mantissa
    HVX_ಸದಿಶ vout   = Q6_Vw_vadd_VwVw(mant_v, ಮುಖವಾಡ_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_ಸದಿಶ neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    ಹಿಂತಿರುಗಿ (vout);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_floor_f32(HVX_ಸದಿಶ ಒಳ_vec) {
    HVX_ಸದಿಶ ಮುಖವಾಡ_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_ಸದಿಶ ಮುಖವಾಡ_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_ಸದಿಶ const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_ಸದಿಶ const_zero_v   = Q6_V_vzero();
    HVX_ಸದಿಶ const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_ಸದಿಶ_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, ಒಳ_vec);

    HVX_ಸದಿಶ expval_v = ಒಳ_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_ಸದಿಶ_Pred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_ಸದಿಶ_Pred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_ಸದಿಶ_Pred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, ಒಳ_vec, const_zero_v);
    HVX_ಸದಿಶ_Pred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, ಒಳ_vec);

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
    ಮುಖವಾಡ_mant_v >>= expval_v;
    HVX_ಸದಿಶ neg_addin_v    = ಮುಖವಾಡ_impl_v >> expval_v;
    HVX_ಸದಿಶ vout_neg_addin = Q6_Vw_vadd_VwVw(ಒಳ_vec, neg_addin_v);
    HVX_ಸದಿಶ vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, ಒಳ_vec);

    HVX_ಸದಿಶ     ಮುಖವಾಡ_chk_v = Q6_V_vand_VV(ಒಳ_vec, ಮುಖವಾಡ_mant_v);  // chk if bits set
    HVX_ಸದಿಶ_Pred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, ಮುಖವಾಡ_chk_v);

    HVX_ಸದಿಶ not_ಮುಖವಾಡ_v = Q6_V_vnot_V(ಮುಖವಾಡ_mant_v);        // frac bits to clear
    HVX_ಸದಿಶ vfrfloor_v = Q6_V_vand_VV(vout, not_ಮುಖವಾಡ_v);  // clear frac bits

    vout = ಒಳ_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, ಒಳ_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    ಹಿಂತಿರುಗಿ vout;
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_FLOOR_H */
