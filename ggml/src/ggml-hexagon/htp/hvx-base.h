#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_ਅਧਾਰ_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_ਅਧਾਰ_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-types.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_vmem(A)   *((HVX_ਸਦਿਸ਼ *)(A))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_vmemu(A)  *((HVX_UVector *)(A))

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_vec_store_u(ਖਾਲੀ * ਸੀਮਿਤ dst, uint32_t n, HVX_ਸਦਿਸ਼ v) {
    // Rotate as needed.
    v = Q6_V_vlalign_VVR(v, v, (size_t) dst);

    uint32_t ਖੱਬਾ_off  = (size_t) dst & 127;
    uint32_t ਸੱਜਾ_off = ਖੱਬਾ_off + n;

    HVX_ਸਦਿਸ਼_Pred ql_not = Q6_Q_vsetq_R((size_t) dst);
    HVX_ਸਦਿਸ਼_Pred qr     = Q6_Q_vsetq2_R(ਸੱਜਾ_off);

    ਜੇ (ਸੱਜਾ_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_ਸਦਿਸ਼ *) dst + 1, v);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(v, v);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_ਸਦਿਸ਼ *) dst, v);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_vec_store_a(ਖਾਲੀ * ਸੀਮਿਤ dst, uint32_t n, HVX_ਸਦਿਸ਼ v) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    HVX_ਸਦਿਸ਼_Pred m = Q6_Q_or_QQn(Q6_Q_vsetq_R((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst), Q6_Q_vsetq2_R(n));
    Q6_vmem_QnRIV(m, (HVX_ਸਦਿਸ਼ *) dst, v);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_splat_f32(ਦਸ਼ਮਲਵ v) {
    ਸੰਘ { ਦਸ਼ਮਲਵ  f; uint32_t i; } u = { .f = v };
    ਵਾਪਸ Q6_V_vsplat_R(u.i);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_splat_f16(_Float16 v) {
    ਸੰਘ { __fp16 f; uint16_t i; } u = { .f = v };
    ਵਾਪਸ Q6_Vh_vsplat_R(u.i);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_repl4(HVX_ਸਦਿਸ਼ v) {
    // vdelta control to replicate first 4 bytes across all elements
    ਸਥਿਰ ਅਚਲ uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };

    HVX_ਸਦਿਸ਼ ctrl = *(HVX_ਸਦਿਸ਼ *) repl;
    ਵਾਪਸ Q6_V_vdelta_VV(v, ctrl);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hvx_vec_get_f32(HVX_ਸਦਿਸ਼ v) {
    ਦਸ਼ਮਲਵ __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    ਵਾਪਸ x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t hvx_vec_get_i32(HVX_ਸਦਿਸ਼ v) {
    int32_t __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    ਵਾਪਸ x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ _Float16 hvx_vec_get_f16(HVX_ਸਦਿਸ਼ v) {
    _Float16 __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 2, v);
    ਵਾਪਸ x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_abs_f16(HVX_ਸਦਿਸ਼ v) {
    // abs by clearing the fp16 sign bit
    HVX_ਸਦਿਸ਼ ਨਕਾਬ = Q6_Vh_vsplat_R(0x7fff);
    ਵਾਪਸ Q6_V_vand_VV(v, ਨਕਾਬ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_neg_f16(HVX_ਸਦਿਸ਼ v) {
    // neg by setting the fp16 sign bit
    HVX_ਸਦਿਸ਼ ਨਕਾਬ = Q6_Vh_vsplat_R(0x8000);
    ਵਾਪਸ Q6_V_vxor_VV(v, ਨਕਾਬ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_abs_f32(HVX_ਸਦਿਸ਼ v) {
    // abs by clearing the fp32 sign bit
    HVX_ਸਦਿਸ਼ ਨਕਾਬ = Q6_V_vsplat_R(0x7fffffff);
    ਵਾਪਸ Q6_V_vand_VV(v, ਨਕਾਬ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_neg_f32(HVX_ਸਦਿਸ਼ v) {
#ਜੇ __HVX_ARCH__ > 75
    ਵਾਪਸ Q6_Vsf_vfneg_Vsf(v);
#ਨਹੀਂਤਾਂ
    // neg by setting the fp32 sign bit
    HVX_ਸਦਿਸ਼ ਨਕਾਬ = Q6_V_vsplat_R(0x80000000);
    ਵਾਪਸ Q6_V_vxor_VV(v, ਨਕਾਬ);
#ਜੇ_ਅੰਤ  // __HVX_ARCH__ > 75
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_ਕਦਮ_f32(HVX_ਸਦਿਸ਼ v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ one  = hvx_vec_splat_f32(1.0f);
    HVX_ਸਦਿਸ਼_Pred q = Q6_Q_vcmp_gt_VsfVsf(v, zero);
    ਵਾਪਸ Q6_V_vmux_QVV(q, one, zero);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_ਕਦਮ_f16(HVX_ਸਦਿਸ਼ v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();
    ਅਚਲ HVX_ਸਦਿਸ਼ one  = hvx_vec_splat_f16((_Float16) 1.0f);
    HVX_ਸਦਿਸ਼_Pred q = Q6_Q_vcmp_gt_VhfVhf(v, zero);
    ਵਾਪਸ Q6_V_vmux_QVV(q, one, zero);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pred hvx_vec_is_nan_f16(HVX_ਸਦਿਸ਼ v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ vnan_exp  = Q6_Vh_vsplat_R(0x7C00);
    ਅਚਲ HVX_ਸਦਿਸ਼ vnan_frac = Q6_Vh_vsplat_R(0x7FFF);

    // get pred of which are NaN, i.e., exponent bits all 1s and fraction bits non 0s
    HVX_ਸਦਿਸ਼_Pred p_exp  = Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_exp), vnan_exp);
    HVX_ਸਦਿਸ਼_Pred p_frac = Q6_Q_not_Q(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_frac), vnan_exp));
    ਵਾਪਸ Q6_Q_and_QQ(p_exp, p_frac);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(HVX_ਸਦਿਸ਼ v0, HVX_ਸਦਿਸ਼ v1) {
#ਜੇ __HVX_ARCH__ >= 81
    HVX_ਸਦਿਸ਼ q0 = Q6_Vqf32_equals_Vsf(v0);
    HVX_ਸਦਿਸ਼ q1 = Q6_Vqf32_equals_Vsf(v1);
#ਨਹੀਂਤਾਂ
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ q0 = Q6_Vqf32_vadd_VsfVsf(v0, zero);
    HVX_ਸਦਿਸ਼ q1 = Q6_Vqf32_vadd_VsfVsf(v1, zero);
#ਜੇ_ਅੰਤ
    ਵਾਪਸ Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(HVX_ਸਦਿਸ਼ v0, HVX_ਸਦਿਸ਼ v1) {
    ਵਾਪਸ Q6_Vh_vdeal_Vh(hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1));
}

#ਜੇ __HVX_ARCH__ >= 79
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(HVX_ਸਦਿਸ਼ v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f16(1.0);
    HVX_ਸਦਿਸ਼_Pair p = Q6_Wsf_vmpy_VhfVhf(v, one);
    ਵਾਪਸ Q6_W_vcombine_VV(Q6_V_hi_W(p), Q6_V_lo_W(p));
}
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(HVX_ਸਦਿਸ਼ v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f16(1.0);
    HVX_ਸਦਿਸ਼_Pair p = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    ਵਾਪਸ Q6_W_vcombine_VV(Q6_V_hi_W(p), Q6_V_lo_W(p));
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(HVX_ਸਦਿਸ਼ v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f16(1.0);
    HVX_ਸਦਿਸ਼_Pair p = Q6_Wqf32_vmpy_VhfVhf(v, one);
    ਵਾਪਸ Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)));
}
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(HVX_ਸਦਿਸ਼ v) {
    ਅਚਲ HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f16(1.0);
    HVX_ਸਦਿਸ਼_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    ਵਾਪਸ Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)));
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_i16_from_hf_rnd_sat(HVX_ਸਦਿਸ਼ vin) {
    // This looks complicated.
    // Ideally should just be Q6_Vh_equals_Vhf(vin)
    // but that instruction does not do proper rounding.

    // convert to qf32, multiplying by 1.0 in the process.
    HVX_ਸਦਿਸ਼_Pair v32 = Q6_Wqf32_vmpy_VhfVhf(vin, Q6_Vh_vsplat_R(0x3C00));

    // 'in-range' values are +/32752.
    // add 192K to it, convert to sf
    HVX_ਸਦਿਸ਼ v192K = Q6_V_vsplat_R(0x48400000);
    HVX_ਸਦਿਸ਼ vsf_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(v32), v192K));
    HVX_ਸਦਿਸ਼ vsf_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(v32), v192K));

    // for in-range cases, result is {163858... 229360} so the exponent is always 144.
    // if we extract bits 21..0 as a signed quantity, and round 6 bits off, that will be the answer.
    // Start by <<10 to get the final 'sign' bit in bit 15...
    vsf_0 = Q6_Vw_vasl_VwR(vsf_0, 10);
    vsf_1 = Q6_Vw_vasl_VwR(vsf_1, 10);

    // now round down to 16
    ਵਾਪਸ Q6_Vh_vround_VwVw_sat(vsf_1, vsf_0);
}

#ਜੇ __HVX_ARCH__ < 79

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_vec_mpyacc_f32_f16(HVX_ਸਦਿਸ਼_Pair acc, HVX_ਸਦਿਸ਼ x, HVX_ਸਦਿਸ਼ y)
{
    HVX_ਸਦਿਸ਼_Pair m = Q6_Wqf32_vmpy_VhfVhf(x, y);
    HVX_ਸਦਿਸ਼ a0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(m), Q6_V_lo_W(acc)));
    HVX_ਸਦਿਸ਼ a1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(m), Q6_V_hi_W(acc)));
    ਵਾਪਸ Q6_W_vcombine_VV(a1, a0);
}

#ਨਹੀਂਤਾਂ

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_vec_mpyacc_f32_f16(HVX_ਸਦਿਸ਼_Pair acc, HVX_ਸਦਿਸ਼ x, HVX_ਸਦਿਸ਼ y)
{
    ਵਾਪਸ Q6_Wsf_vmpyacc_WsfVhfVhf(acc, x, y);
}

#ਜੇ_ਅੰਤ

#ਜੇ __HVX_ARCH__ < 79

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_ਜੋੜੋ_f16_f16(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b)
{
    ਅਚਲ HVX_ਸਦਿਸ਼ negone = Q6_Vh_vsplat_R(0xBC00); // -1.0 in IEEE FP16
    ਅਚਲ HVX_ਸਦਿਸ਼ one    = Q6_Vh_vsplat_R(0x3C00); //  1.0 in IEEE FP16
    HVX_ਸਦਿਸ਼_Pair a_p = Q6_Wqf32_vmpy_VhfVhf(a, one);
    HVX_ਸਦਿਸ਼_Pair b_p = Q6_Wqf32_vmpy_VhfVhf(b, negone);
    HVX_ਸਦਿਸ਼ a0 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(a_p), Q6_V_lo_W(b_p));
    HVX_ਸਦਿਸ਼ a1 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(a_p), Q6_V_hi_W(b_p));
    ਵਾਪਸ Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1, a0));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_sub_f16_f16(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b)
{
    ਅਚਲ HVX_ਸਦਿਸ਼ negone = Q6_Vh_vsplat_R(0xBC00); // -1.0 in IEEE FP16
    ਅਚਲ HVX_ਸਦਿਸ਼ one    = Q6_Vh_vsplat_R(0x3C00); //  1.0 in IEEE FP16
    HVX_ਸਦਿਸ਼_Pair a_p = Q6_Wqf32_vmpy_VhfVhf(a, one);
    HVX_ਸਦਿਸ਼_Pair b_p = Q6_Wqf32_vmpy_VhfVhf(b, negone);
    HVX_ਸਦਿਸ਼ a0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(a_p), Q6_V_lo_W(b_p));
    HVX_ਸਦਿਸ਼ a1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(a_p), Q6_V_hi_W(b_p));
    ਵਾਪਸ Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1, a0));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_mul_f16_f16(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b)
{
    ਵਾਪਸ Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_ਜੋੜੋ_f32_f32(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_sub_f32_f32(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_mul_f32_f32(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

#ਨਹੀਂਤਾਂ

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_ਜੋੜੋ_f16_f16(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b)
{
    ਵਾਪਸ Q6_Vhf_vadd_VhfVhf(a, b);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_sub_f16_f16(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b)
{
    ਵਾਪਸ Q6_Vhf_vsub_VhfVhf(a, b);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_mul_f16_f16(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b)
{
    ਵਾਪਸ Q6_Vhf_vmpy_VhfVhf(a, b);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_ਜੋੜੋ_f32_f32(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_vadd_VsfVsf(a, b);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_sub_f32_f32(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_vsub_VsfVsf(a, b);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_mul_f32_f32(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_vmpy_VsfVsf(a, b);
}

#ਜੇ_ਅੰਤ // __HVX_ARCH__ < 79

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_ਲੱਦੋ_act_tile(ਅਚਲ uint8_t * y_q, uint32_t kt, HVX_ਸਦਿਸ਼ * v_act_all) {
    ਜੇ (kt % 4 == 0) {
        *v_act_all = hvx_vmem(y_q + kt * 32);
        ਵਾਪਸ *v_act_all;
    } ਨਹੀਂਤਾਂ ਜੇ (kt % 4 == 1) {
        ਵਾਪਸ Q6_V_vror_VR(*v_act_all, 32);
    } ਨਹੀਂਤਾਂ ਜੇ (kt % 4 == 2) {
        ਵਾਪਸ Q6_V_vror_VR(*v_act_all, 64);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ Q6_V_vror_VR(*v_act_all, 96);
    }
}

// Full tile, in place. tmp: 32 vectors scratch.
// Use when result must stay in m. Slowest (spills).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_transpose_32x32_w_shuff_inplace(HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ m, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ tmp) {
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 4) {
        HVX_ਸਦਿਸ਼_Pair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_ਸਦਿਸ਼_Pair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 8) {
        ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
            HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ਲਈ (ਪੂਰਨ b = 0; b < 32; b += 16) {
        ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
            HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// Rows [0, nrows) of the transpose of src[0, ncols) to dst. src is clobbered, must not alias dst.
// Use when not in VTCM or for full tiles.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_transpose_32x32_w_shuff(uint8_t * ਸੀਮਿਤ dst, uint32_t dst_stride, HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ src,
                                               uint32_t nrows, uint32_t ncols) {
    ਲਈ (uint32_t g = 0; g < ncols; g += 8) {
        HVX_ਸਦਿਸ਼ * s = src + g;

        HVX_ਸਦਿਸ਼_Pair a0 = Q6_W_vshuff_VVR(s[1], s[0], -4);
        HVX_ਸਦਿਸ਼_Pair a1 = Q6_W_vshuff_VVR(s[3], s[2], -4);
        HVX_ਸਦਿਸ਼_Pair a2 = Q6_W_vshuff_VVR(s[5], s[4], -4);
        HVX_ਸਦਿਸ਼_Pair a3 = Q6_W_vshuff_VVR(s[7], s[6], -4);

        HVX_ਸਦਿਸ਼_Pair b0 = Q6_W_vshuff_VVR(Q6_V_lo_W(a1), Q6_V_lo_W(a0), -8);
        HVX_ਸਦਿਸ਼_Pair b1 = Q6_W_vshuff_VVR(Q6_V_hi_W(a1), Q6_V_hi_W(a0), -8);
        HVX_ਸਦਿਸ਼_Pair b2 = Q6_W_vshuff_VVR(Q6_V_lo_W(a3), Q6_V_lo_W(a2), -8);
        HVX_ਸਦਿਸ਼_Pair b3 = Q6_W_vshuff_VVR(Q6_V_hi_W(a3), Q6_V_hi_W(a2), -8);

        HVX_ਸਦਿਸ਼_Pair c0 = Q6_W_vshuff_VVR(Q6_V_lo_W(b2), Q6_V_lo_W(b0), -16);
        HVX_ਸਦਿਸ਼_Pair c1 = Q6_W_vshuff_VVR(Q6_V_hi_W(b2), Q6_V_hi_W(b0), -16);
        HVX_ਸਦਿਸ਼_Pair c2 = Q6_W_vshuff_VVR(Q6_V_lo_W(b3), Q6_V_lo_W(b1), -16);
        HVX_ਸਦਿਸ਼_Pair c3 = Q6_W_vshuff_VVR(Q6_V_hi_W(b3), Q6_V_hi_W(b1), -16);

        s[0] = Q6_V_lo_W(c0); s[1] = Q6_V_hi_W(c0);
        s[2] = Q6_V_lo_W(c1); s[3] = Q6_V_hi_W(c1);
        s[4] = Q6_V_lo_W(c2); s[5] = Q6_V_hi_W(c2);
        s[6] = Q6_V_lo_W(c3); s[7] = Q6_V_hi_W(c3);
    }

    ਲਈ (uint32_t k = 0; 4 * k < nrows; k++) {
        HVX_ਸਦਿਸ਼_Pair d0 = Q6_W_vshuff_VVR(src[k + 8],  src[k],      -32);
        HVX_ਸਦਿਸ਼_Pair d1 = Q6_W_vshuff_VVR(src[k + 24], src[k + 16], -32);

        HVX_ਸਦਿਸ਼_Pair e0 = Q6_W_vshuff_VVR(Q6_V_lo_W(d1), Q6_V_lo_W(d0), -64);
        HVX_ਸਦਿਸ਼_Pair e1 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1), Q6_V_hi_W(d0), -64);

        uint8_t * d = dst + 4 * k * dst_stride;
        *(HVX_ਸਦਿਸ਼ *) (d + 0 * dst_stride) = Q6_V_lo_W(e0);
        *(HVX_ਸਦਿਸ਼ *) (d + 1 * dst_stride) = Q6_V_hi_W(e0);
        *(HVX_ਸਦਿਸ਼ *) (d + 2 * dst_stride) = Q6_V_lo_W(e1);
        *(HVX_ਸਦਿਸ਼ *) (d + 3 * dst_stride) = Q6_V_hi_W(e1);
    }
}

// Specialized store that will follow outstanding scatters or gathers to make sure they complete
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_gather_sync(ਖਾਲੀ * vtcm_addr) {
    asm ਅਸਥਿਰ("vmem(%0+#0):scatter_release\n"
                 "v0 = vmem(%0+#0)\n" :: "r"(vtcm_addr) : "v0", "memory");
}

// Lane k = k * stride. Offsets for the _gather transposes.
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_gather_offsets_w(uint32_t stride) {
    int32_t offsets[32] __attribute__((aligned(128)));
    ਲਈ (ਪੂਰਨ k = 0; k < 32; k++) {
        offsets[k] = k * stride;
    }
    ਵਾਪਸ *(HVX_ਸਦਿਸ਼ *) offsets;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_gather_offsets_h(uint32_t stride) {
    int16_t offsets[64] __attribute__((aligned(128)));
    ਲਈ (ਪੂਰਨ k = 0; k < 64; k++) {
        offsets[k] = k * stride;
    }
    ਵਾਪਸ *(HVX_ਸਦਿਸ਼ *) offsets;
}

// Rows [0, nrows) of the transpose of ncols src vectors (src_stride apart) to dst.
// src, dst in VTCM. Async: hvx_gather_sync() before use.
// Use for partial tiles (small ncols). Fewest packets.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_transpose_32x32_w_gather(uint8_t * ਸੀਮਿਤ dst, uint32_t dst_stride, ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                                uint32_t src_stride, HVX_ਸਦਿਸ਼ offsets, uint32_t nrows, uint32_t ncols) {
    ਅਚਲ uint32_t mu = ncols * src_stride;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t r = 0; r < nrows; r++) {
        Q6_vgather_ARMVw((HVX_ਸਦਿਸ਼ *) (dst + r * dst_stride), (size_t) (src + r * ਆਕਾਰ(int32_t)), mu, offsets);
    }
}

// 16-bit version of hvx_transpose_32x32_w_gather.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_transpose_64x64_h_gather(uint8_t * ਸੀਮਿਤ dst, uint32_t dst_stride, ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                                uint32_t src_stride, HVX_ਸਦਿਸ਼ offsets, uint32_t nrows, uint32_t ncols) {
    ਅਚਲ uint32_t mu = ncols * src_stride;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t r = 0; r < nrows; r++) {
        Q6_vgather_ARMVh((HVX_ਸਦਿਸ਼ *) (dst + r * dst_stride), (size_t) (src + r * ਆਕਾਰ(int16_t)), mu, offsets);
    }
}

#ਜੇ_ਅੰਤ /* HVX_BASE_H */
