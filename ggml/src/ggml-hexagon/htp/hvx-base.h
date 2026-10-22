#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_ತಳ_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_ತಳ_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <assert.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_vmem(A)   *((HVX_ಸದಿಶ *)(A))
#ವ್ಯಾಖ್ಯಾನಿಸು hvx_vmemu(A)  *((HVX_UVector *)(A))

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_vec_store_u(ಶೂನ್ಯ * ಸೀಮಿತ dst, uint32_t n, HVX_ಸದಿಶ v) {
    // Rotate as needed.
    v = Q6_V_vlalign_VVR(v, v, (size_t) dst);

    uint32_t ಎಡ_off  = (size_t) dst & 127;
    uint32_t ಬಲ_off = ಎಡ_off + n;

    HVX_ಸದಿಶ_Pred ql_not = Q6_Q_vsetq_R((size_t) dst);
    HVX_ಸದಿಶ_Pred qr     = Q6_Q_vsetq2_R(ಬಲ_off);

    ಆದರೆ (ಬಲ_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_ಸದಿಶ *) dst + 1, v);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(v, v);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_ಸದಿಶ *) dst, v);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_vec_store_a(ಶೂನ್ಯ * ಸೀಮಿತ dst, uint32_t n, HVX_ಸದಿಶ v) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    HVX_ಸದಿಶ_Pred m = Q6_Q_or_QQn(Q6_Q_vsetq_R((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst), Q6_Q_vsetq2_R(n));
    Q6_vmem_QnRIV(m, (HVX_ಸದಿಶ *) dst, v);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_splat_f32(ದಶಮಾಂಶ v) {
    ಒಕ್ಕೂಟ { ದಶಮಾಂಶ  f; uint32_t i; } u = { .f = v };
    ಹಿಂತಿರುಗಿ Q6_V_vsplat_R(u.i);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_splat_f16(_Float16 v) {
    ಒಕ್ಕೂಟ { __fp16 f; uint16_t i; } u = { .f = v };
    ಹಿಂತಿರುಗಿ Q6_Vh_vsplat_R(u.i);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_repl4(HVX_ಸದಿಶ v) {
    // vdelta control to replicate first 4 bytes across all elements
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };

    HVX_ಸದಿಶ ctrl = *(HVX_ಸದಿಶ *) repl;
    ಹಿಂತಿರುಗಿ Q6_V_vdelta_VV(v, ctrl);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hvx_vec_get_f32(HVX_ಸದಿಶ v) {
    ದಶಮಾಂಶ __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t hvx_vec_get_i32(HVX_ಸದಿಶ v) {
    int32_t __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ _Float16 hvx_vec_get_f16(HVX_ಸದಿಶ v) {
    _Float16 __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 2, v);
    ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_abs_f16(HVX_ಸದಿಶ v) {
    // abs by clearing the fp16 sign bit
    HVX_ಸದಿಶ ಮುಖವಾಡ = Q6_Vh_vsplat_R(0x7fff);
    ಹಿಂತಿರುಗಿ Q6_V_vand_VV(v, ಮುಖವಾಡ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_neg_f16(HVX_ಸದಿಶ v) {
    // neg by setting the fp16 sign bit
    HVX_ಸದಿಶ ಮುಖವಾಡ = Q6_Vh_vsplat_R(0x8000);
    ಹಿಂತಿರುಗಿ Q6_V_vxor_VV(v, ಮುಖವಾಡ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_abs_f32(HVX_ಸದಿಶ v) {
    // abs by clearing the fp32 sign bit
    HVX_ಸದಿಶ ಮುಖವಾಡ = Q6_V_vsplat_R(0x7fffffff);
    ಹಿಂತಿರುಗಿ Q6_V_vand_VV(v, ಮುಖವಾಡ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_neg_f32(HVX_ಸದಿಶ v) {
#ಆದರೆ __HVX_ARCH__ > 75
    ಹಿಂತಿರುಗಿ Q6_Vsf_vfneg_Vsf(v);
#ಇಲ್ಲದಿದ್ದರೆ
    // neg by setting the fp32 sign bit
    HVX_ಸದಿಶ ಮುಖವಾಡ = Q6_V_vsplat_R(0x80000000);
    ಹಿಂತಿರುಗಿ Q6_V_vxor_VV(v, ಮುಖವಾಡ);
#ಆದರೆ_ಅಂತ್ಯ  // __HVX_ARCH__ > 75
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_ಹೆಜ್ಜೆ_f32(HVX_ಸದಿಶ v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one  = hvx_vec_splat_f32(1.0f);
    HVX_ಸದಿಶ_Pred q = Q6_Q_vcmp_gt_VsfVsf(v, zero);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(q, one, zero);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_ಹೆಜ್ಜೆ_f16(HVX_ಸದಿಶ v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one  = hvx_vec_splat_f16((_Float16) 1.0f);
    HVX_ಸದಿಶ_Pred q = Q6_Q_vcmp_gt_VhfVhf(v, zero);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(q, one, zero);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pred hvx_vec_is_nan_f16(HVX_ಸದಿಶ v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vnan_exp  = Q6_Vh_vsplat_R(0x7C00);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vnan_frac = Q6_Vh_vsplat_R(0x7FFF);

    // get pred of which are NaN, i.e., exponent bits all 1s and fraction bits non 0s
    HVX_ಸದಿಶ_Pred p_exp  = Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_exp), vnan_exp);
    HVX_ಸದಿಶ_Pred p_frac = Q6_Q_not_Q(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_frac), vnan_exp));
    ಹಿಂತಿರುಗಿ Q6_Q_and_QQ(p_exp, p_frac);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_f32_ಗಮ್ಯ_f16_shuff(HVX_ಸದಿಶ v0, HVX_ಸದಿಶ v1) {
#ಆದರೆ __HVX_ARCH__ >= 81
    HVX_ಸದಿಶ q0 = Q6_Vqf32_equals_Vsf(v0);
    HVX_ಸದಿಶ q1 = Q6_Vqf32_equals_Vsf(v1);
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = Q6_V_vzero();
    HVX_ಸದಿಶ q0 = Q6_Vqf32_vadd_VsfVsf(v0, zero);
    HVX_ಸದಿಶ q1 = Q6_Vqf32_vadd_VsfVsf(v1, zero);
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_f32_ಗಮ್ಯ_f16(HVX_ಸದಿಶ v0, HVX_ಸದಿಶ v1) {
    ಹಿಂತಿರುಗಿ Q6_Vh_vdeal_Vh(hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1));
}

#ಆದರೆ __HVX_ARCH__ >= 79
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_vec_f16_ಗಮ್ಯ_f32_shuff(HVX_ಸದಿಶ v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one = hvx_vec_splat_f16(1.0);
    HVX_ಸದಿಶ_Pair p = Q6_Wsf_vmpy_VhfVhf(v, one);
    ಹಿಂತಿರುಗಿ Q6_W_vcombine_VV(Q6_V_hi_W(p), Q6_V_lo_W(p));
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_vec_f16_ಗಮ್ಯ_f32(HVX_ಸದಿಶ v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one = hvx_vec_splat_f16(1.0);
    HVX_ಸದಿಶ_Pair p = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    ಹಿಂತಿರುಗಿ Q6_W_vcombine_VV(Q6_V_hi_W(p), Q6_V_lo_W(p));
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_vec_f16_ಗಮ್ಯ_f32_shuff(HVX_ಸದಿಶ v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one = hvx_vec_splat_f16(1.0);
    HVX_ಸದಿಶ_Pair p = Q6_Wqf32_vmpy_VhfVhf(v, one);
    ಹಿಂತಿರುಗಿ Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)));
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_vec_f16_ಗಮ್ಯ_f32(HVX_ಸದಿಶ v) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one = hvx_vec_splat_f16(1.0);
    HVX_ಸದಿಶ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    ಹಿಂತಿರುಗಿ Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)));
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_i16_from_hf_rnd_sat(HVX_ಸದಿಶ vin) {
    // This looks complicated.
    // Ideally should just be Q6_Vh_equals_Vhf(vin)
    // but that instruction does not do proper rounding.

    // convert to qf32, multiplying by 1.0 in the process.
    HVX_ಸದಿಶ_Pair v32 = Q6_Wqf32_vmpy_VhfVhf(vin, Q6_Vh_vsplat_R(0x3C00));

    // 'in-range' values are +/32752.
    // add 192K to it, convert to sf
    HVX_ಸದಿಶ v192K = Q6_V_vsplat_R(0x48400000);
    HVX_ಸದಿಶ vsf_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(v32), v192K));
    HVX_ಸದಿಶ vsf_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(v32), v192K));

    // for in-range cases, result is {163858... 229360} so the exponent is always 144.
    // if we extract bits 21..0 as a signed quantity, and round 6 bits off, that will be the answer.
    // Start by <<10 to get the final 'sign' bit in bit 15...
    vsf_0 = Q6_Vw_vasl_VwR(vsf_0, 10);
    vsf_1 = Q6_Vw_vasl_VwR(vsf_1, 10);

    // now round down to 16
    ಹಿಂತಿರುಗಿ Q6_Vh_vround_VwVw_sat(vsf_1, vsf_0);
}

#ಆದರೆ __HVX_ARCH__ < 79

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_vec_mpyacc_f32_f16(HVX_ಸದಿಶ_Pair acc, HVX_ಸದಿಶ x, HVX_ಸದಿಶ y)
{
    HVX_ಸದಿಶ_Pair m = Q6_Wqf32_vmpy_VhfVhf(x, y);
    HVX_ಸದಿಶ a0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(m), Q6_V_lo_W(acc)));
    HVX_ಸದಿಶ a1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(m), Q6_V_hi_W(acc)));
    ಹಿಂತಿರುಗಿ Q6_W_vcombine_VV(a1, a0);
}

#ಇಲ್ಲದಿದ್ದರೆ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_vec_mpyacc_f32_f16(HVX_ಸದಿಶ_Pair acc, HVX_ಸದಿಶ x, HVX_ಸದಿಶ y)
{
    ಹಿಂತಿರುಗಿ Q6_Wsf_vmpyacc_WsfVhfVhf(acc, x, y);
}

#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ __HVX_ARCH__ < 79

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_ಕೂಡಿಸು_f16_f16(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b)
{
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ negone = Q6_Vh_vsplat_R(0xBC00); // -1.0 in IEEE FP16
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one    = Q6_Vh_vsplat_R(0x3C00); //  1.0 in IEEE FP16
    HVX_ಸದಿಶ_Pair a_p = Q6_Wqf32_vmpy_VhfVhf(a, one);
    HVX_ಸದಿಶ_Pair b_p = Q6_Wqf32_vmpy_VhfVhf(b, negone);
    HVX_ಸದಿಶ a0 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(a_p), Q6_V_lo_W(b_p));
    HVX_ಸದಿಶ a1 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(a_p), Q6_V_hi_W(b_p));
    ಹಿಂತಿರುಗಿ Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1, a0));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_sub_f16_f16(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b)
{
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ negone = Q6_Vh_vsplat_R(0xBC00); // -1.0 in IEEE FP16
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one    = Q6_Vh_vsplat_R(0x3C00); //  1.0 in IEEE FP16
    HVX_ಸದಿಶ_Pair a_p = Q6_Wqf32_vmpy_VhfVhf(a, one);
    HVX_ಸದಿಶ_Pair b_p = Q6_Wqf32_vmpy_VhfVhf(b, negone);
    HVX_ಸದಿಶ a0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(a_p), Q6_V_lo_W(b_p));
    HVX_ಸದಿಶ a1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(a_p), Q6_V_hi_W(b_p));
    ಹಿಂತಿರುಗಿ Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1, a0));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_mul_f16_f16(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b)
{
    ಹಿಂತಿರುಗಿ Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_ಕೂಡಿಸು_f32_f32(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_sub_f32_f32(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_mul_f32_f32(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

#ಇಲ್ಲದಿದ್ದರೆ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_ಕೂಡಿಸು_f16_f16(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b)
{
    ಹಿಂತಿರುಗಿ Q6_Vhf_vadd_VhfVhf(a, b);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_sub_f16_f16(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b)
{
    ಹಿಂತಿರುಗಿ Q6_Vhf_vsub_VhfVhf(a, b);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_mul_f16_f16(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b)
{
    ಹಿಂತಿರುಗಿ Q6_Vhf_vmpy_VhfVhf(a, b);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_ಕೂಡಿಸು_f32_f32(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_vadd_VsfVsf(a, b);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_sub_f32_f32(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_vsub_VsfVsf(a, b);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_mul_f32_f32(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_vmpy_VsfVsf(a, b);
}

#ಆದರೆ_ಅಂತ್ಯ // __HVX_ARCH__ < 79

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_ತುಂಬು_act_tile(ಸ್ಥಿರಾಂಕ uint8_t * y_q, uint32_t kt, HVX_ಸದಿಶ * v_act_all) {
    ಆದರೆ (kt % 4 == 0) {
        *v_act_all = hvx_vmem(y_q + kt * 32);
        ಹಿಂತಿರುಗಿ *v_act_all;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kt % 4 == 1) {
        ಹಿಂತಿರುಗಿ Q6_V_vror_VR(*v_act_all, 32);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kt % 4 == 2) {
        ಹಿಂತಿರುಗಿ Q6_V_vror_VR(*v_act_all, 64);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ Q6_V_vror_VR(*v_act_all, 96);
    }
}

// Full tile, in place. tmp: 32 vectors scratch.
// Use when result must stay in m. Slowest (spills).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_transpose_32x32_w_shuff_inplace(HVX_ಸದಿಶ * ಸೀಮಿತ m, HVX_ಸದಿಶ * ಸೀಮಿತ tmp) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 4) {
        HVX_ಸದಿಶ_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_ಸದಿಶ_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 8) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i) {
            HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < 32; b += 16) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i) {
            HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; ++i) {
        HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// Rows [0, nrows) of the transpose of src[0, ncols) to dst. src is clobbered, must not alias dst.
// Use when not in VTCM or for full tiles.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_transpose_32x32_w_shuff(uint8_t * ಸೀಮಿತ dst, uint32_t dst_stride, HVX_ಸದಿಶ * ಸೀಮಿತ src,
                                               uint32_t nrows, uint32_t ncols) {
    ಗಾಗಿ (uint32_t g = 0; g < ncols; g += 8) {
        HVX_ಸದಿಶ * s = src + g;

        HVX_ಸದಿಶ_Pair a0 = Q6_W_vshuff_VVR(s[1], s[0], -4);
        HVX_ಸದಿಶ_Pair a1 = Q6_W_vshuff_VVR(s[3], s[2], -4);
        HVX_ಸದಿಶ_Pair a2 = Q6_W_vshuff_VVR(s[5], s[4], -4);
        HVX_ಸದಿಶ_Pair a3 = Q6_W_vshuff_VVR(s[7], s[6], -4);

        HVX_ಸದಿಶ_Pair b0 = Q6_W_vshuff_VVR(Q6_V_lo_W(a1), Q6_V_lo_W(a0), -8);
        HVX_ಸದಿಶ_Pair b1 = Q6_W_vshuff_VVR(Q6_V_hi_W(a1), Q6_V_hi_W(a0), -8);
        HVX_ಸದಿಶ_Pair b2 = Q6_W_vshuff_VVR(Q6_V_lo_W(a3), Q6_V_lo_W(a2), -8);
        HVX_ಸದಿಶ_Pair b3 = Q6_W_vshuff_VVR(Q6_V_hi_W(a3), Q6_V_hi_W(a2), -8);

        HVX_ಸದಿಶ_Pair c0 = Q6_W_vshuff_VVR(Q6_V_lo_W(b2), Q6_V_lo_W(b0), -16);
        HVX_ಸದಿಶ_Pair c1 = Q6_W_vshuff_VVR(Q6_V_hi_W(b2), Q6_V_hi_W(b0), -16);
        HVX_ಸದಿಶ_Pair c2 = Q6_W_vshuff_VVR(Q6_V_lo_W(b3), Q6_V_lo_W(b1), -16);
        HVX_ಸದಿಶ_Pair c3 = Q6_W_vshuff_VVR(Q6_V_hi_W(b3), Q6_V_hi_W(b1), -16);

        s[0] = Q6_V_lo_W(c0); s[1] = Q6_V_hi_W(c0);
        s[2] = Q6_V_lo_W(c1); s[3] = Q6_V_hi_W(c1);
        s[4] = Q6_V_lo_W(c2); s[5] = Q6_V_hi_W(c2);
        s[6] = Q6_V_lo_W(c3); s[7] = Q6_V_hi_W(c3);
    }

    ಗಾಗಿ (uint32_t k = 0; 4 * k < nrows; k++) {
        HVX_ಸದಿಶ_Pair d0 = Q6_W_vshuff_VVR(src[k + 8],  src[k],      -32);
        HVX_ಸದಿಶ_Pair d1 = Q6_W_vshuff_VVR(src[k + 24], src[k + 16], -32);

        HVX_ಸದಿಶ_Pair e0 = Q6_W_vshuff_VVR(Q6_V_lo_W(d1), Q6_V_lo_W(d0), -64);
        HVX_ಸದಿಶ_Pair e1 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1), Q6_V_hi_W(d0), -64);

        uint8_t * d = dst + 4 * k * dst_stride;
        *(HVX_ಸದಿಶ *) (d + 0 * dst_stride) = Q6_V_lo_W(e0);
        *(HVX_ಸದಿಶ *) (d + 1 * dst_stride) = Q6_V_hi_W(e0);
        *(HVX_ಸದಿಶ *) (d + 2 * dst_stride) = Q6_V_lo_W(e1);
        *(HVX_ಸದಿಶ *) (d + 3 * dst_stride) = Q6_V_hi_W(e1);
    }
}

// Specialized store that will follow outstanding scatters or gathers to make sure they complete
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_gather_sync(ಶೂನ್ಯ * vtcm_addr) {
    asm ಅಸ್ಥಿರ("vmem(%0+#0):scatter_release\n"
                 "v0 = vmem(%0+#0)\n" :: "r"(vtcm_addr) : "v0", "memory");
}

// Lane k = k * stride. Offsets for the _gather transposes.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_gather_offsets_w(uint32_t stride) {
    int32_t offsets[32] __attribute__((aligned(128)));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 32; k++) {
        offsets[k] = k * stride;
    }
    ಹಿಂತಿರುಗಿ *(HVX_ಸದಿಶ *) offsets;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_gather_offsets_h(uint32_t stride) {
    int16_t offsets[64] __attribute__((aligned(128)));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 64; k++) {
        offsets[k] = k * stride;
    }
    ಹಿಂತಿರುಗಿ *(HVX_ಸದಿಶ *) offsets;
}

// Rows [0, nrows) of the transpose of ncols src vectors (src_stride apart) to dst.
// src, dst in VTCM. Async: hvx_gather_sync() before use.
// Use for partial tiles (small ncols). Fewest packets.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_transpose_32x32_w_gather(uint8_t * ಸೀಮಿತ dst, uint32_t dst_stride, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                                uint32_t src_stride, HVX_ಸದಿಶ offsets, uint32_t nrows, uint32_t ncols) {
    ಸ್ಥಿರಾಂಕ uint32_t mu = ncols * src_stride;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t r = 0; r < nrows; r++) {
        Q6_vgather_ARMVw((HVX_ಸದಿಶ *) (dst + r * dst_stride), (size_t) (src + r * ಗಾತ್ರ(int32_t)), mu, offsets);
    }
}

// 16-bit version of hvx_transpose_32x32_w_gather.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_transpose_64x64_h_gather(uint8_t * ಸೀಮಿತ dst, uint32_t dst_stride, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                                uint32_t src_stride, HVX_ಸದಿಶ offsets, uint32_t nrows, uint32_t ncols) {
    ಸ್ಥಿರಾಂಕ uint32_t mu = ncols * src_stride;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t r = 0; r < nrows; r++) {
        Q6_vgather_ARMVh((HVX_ಸದಿಶ *) (dst + r * dst_stride), (size_t) (src + r * ಗಾತ್ರ(int16_t)), mu, offsets);
    }
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_BASE_H */
