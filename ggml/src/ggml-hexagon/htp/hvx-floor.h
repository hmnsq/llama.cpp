#నిర్వచితం_కాకపోతే HVX_FLOOR_H
#నిర్వచించు HVX_FLOOR_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hvx-base.h"

#నిర్వచించు IEEE_VSF_EXPLEN   (8)
#నిర్వచించు IEEE_VSF_EXPBIAS  (127)
#నిర్వచించు IEEE_VSF_EXPMASK  (0xFF)
#నిర్వచించు IEEE_VSF_MANTLEN  (23)
#నిర్వచించు IEEE_VSF_MANTMASK (0x7FFFFF)
#నిర్వచించు IEEE_VSF_MIMPMASK (0x800000)

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_truncate_f32(HVX_సదిశ లోపల_vec) {
    HVX_సదిశ ముసుగు_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_సదిశ ముసుగు_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_సదిశ const_zero_v = Q6_V_vzero();

    HVX_సదిశ_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, లోపల_vec);

    HVX_సదిశ expval_v = లోపల_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_సదిశ_Pred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_సదిశ rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_సదిశ mant_v = లోపల_vec & ముసుగు_mant_v;                  // obtain mantissa
    HVX_సదిశ vout   = Q6_Vw_vadd_VwVw(mant_v, ముసుగు_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_సదిశ neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    తిరిగి (vout);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_floor_f32(HVX_సదిశ లోపల_vec) {
    HVX_సదిశ ముసుగు_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_సదిశ ముసుగు_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_సదిశ const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_సదిశ const_zero_v   = Q6_V_vzero();
    HVX_సదిశ const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_సదిశ_Pred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, లోపల_vec);

    HVX_సదిశ expval_v = లోపల_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_సదిశ_Pred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_సదిశ_Pred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_సదిశ_Pred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, లోపల_vec, const_zero_v);
    HVX_సదిశ_Pred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, లోపల_vec);

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
    ముసుగు_mant_v >>= expval_v;
    HVX_సదిశ neg_addin_v    = ముసుగు_impl_v >> expval_v;
    HVX_సదిశ vout_neg_addin = Q6_Vw_vadd_VwVw(లోపల_vec, neg_addin_v);
    HVX_సదిశ vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, లోపల_vec);

    HVX_సదిశ     ముసుగు_chk_v = Q6_V_vand_VV(లోపల_vec, ముసుగు_mant_v);  // chk if bits set
    HVX_సదిశ_Pred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, ముసుగు_chk_v);

    HVX_సదిశ not_ముసుగు_v = Q6_V_vnot_V(ముసుగు_mant_v);        // frac bits to clear
    HVX_సదిశ vfrfloor_v = Q6_V_vand_VV(vout, not_ముసుగు_v);  // clear frac bits

    vout = లోపల_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, లోపల_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    తిరిగి vout;
}

#అయితే_ముగింపు /* HVX_FLOOR_H */
