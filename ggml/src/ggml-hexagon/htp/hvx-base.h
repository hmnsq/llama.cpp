#ifndef HVX_BASE_H
#define HVX_BASE_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <assert.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hvx-types.h"

#define hvx_vmem(A)   *((HVX_Vector *)(A))
#define hvx_vmemu(A)  *((HVX_UVector *)(A))

स्थिर अन्तःस्थ शून्य hvx_vec_store_u(शून्य * restrict dst, uint32_t n, HVX_Vector v) {
    // Rotate as needed.
    v = Q6_V_vlalign_VVR(v, v, (माप_प्रकार) dst);

    uint32_t बायाँ_off  = (माप_प्रकार) dst & 127;
    uint32_t दायाँ_off = बायाँ_off + n;

    HVX_VectorPred ql_not = Q6_Q_vsetq_R((माप_प्रकार) dst);
    HVX_VectorPred qr     = Q6_Q_vsetq2_R(दायाँ_off);

    यदि (दायाँ_off > 128) {
        Q6_vmem_QRIV(qr, (HVX_Vector *) dst + 1, v);
        // all 1's
        qr = Q6_Q_vcmp_eq_VbVb(v, v);
    }

    ql_not = Q6_Q_or_QQn(ql_not, qr);
    Q6_vmem_QnRIV(ql_not, (HVX_Vector *) dst, v);
}

स्थिर अन्तःस्थ शून्य hvx_vec_store_a(शून्य * restrict dst, uint32_t n, HVX_Vector v) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    HVX_VectorPred m = Q6_Q_or_QQn(Q6_Q_vsetq_R((अचिन्हित दीर्घ) dst), Q6_Q_vsetq2_R(n));
    Q6_vmem_QnRIV(m, (HVX_Vector *) dst, v);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_splat_f32(भिन्न v) {
    सङ्घ { भिन्न  f; uint32_t i; } u = { .f = v };
    प्रत्यागम Q6_V_vsplat_R(u.i);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_splat_f16(_Float16 v) {
    सङ्घ { __fp16 f; uint16_t i; } u = { .f = v };
    प्रत्यागम Q6_Vh_vsplat_R(u.i);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_repl4(HVX_Vector v) {
    // vdelta control to replicate first 4 bytes across all elements
    स्थिर ध्रुव uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };

    HVX_Vector ctrl = *(HVX_Vector *) repl;
    प्रत्यागम Q6_V_vdelta_VV(v, ctrl);
}

स्थिर अन्तःस्थ भिन्न hvx_vec_get_f32(HVX_Vector v) {
    भिन्न __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    प्रत्यागम x;
}

स्थिर अन्तःस्थ int32_t hvx_vec_get_i32(HVX_Vector v) {
    int32_t __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 4, v);
    प्रत्यागम x;
}

स्थिर अन्तःस्थ _Float16 hvx_vec_get_f16(HVX_Vector v) {
    _Float16 __attribute__((aligned(128))) x;
    hvx_vec_store_a(&x, 2, v);
    प्रत्यागम x;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_abs_f16(HVX_Vector v) {
    // abs by clearing the fp16 sign bit
    HVX_Vector mask = Q6_Vh_vsplat_R(0x7fff);
    प्रत्यागम Q6_V_vand_VV(v, mask);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_neg_f16(HVX_Vector v) {
    // neg by setting the fp16 sign bit
    HVX_Vector mask = Q6_Vh_vsplat_R(0x8000);
    प्रत्यागम Q6_V_vxor_VV(v, mask);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_abs_f32(HVX_Vector v) {
    // abs by clearing the fp32 sign bit
    HVX_Vector mask = Q6_V_vsplat_R(0x7fffffff);
    प्रत्यागम Q6_V_vand_VV(v, mask);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_neg_f32(HVX_Vector v) {
#यदि __HVX_ARCH__ > 75
    प्रत्यागम Q6_Vsf_vfneg_Vsf(v);
#अन्यथा
    // neg by setting the fp32 sign bit
    HVX_Vector mask = Q6_V_vsplat_R(0x80000000);
    प्रत्यागम Q6_V_vxor_VV(v, mask);
#endif  // __HVX_ARCH__ > 75
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_step_f32(HVX_Vector v) {
    ध्रुव HVX_Vector zero = Q6_V_vzero();
    ध्रुव HVX_Vector one  = hvx_vec_splat_f32(1.0f);
    HVX_VectorPred q = Q6_Q_vcmp_gt_VsfVsf(v, zero);
    प्रत्यागम Q6_V_vmux_QVV(q, one, zero);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_step_f16(HVX_Vector v) {
    ध्रुव HVX_Vector zero = Q6_V_vzero();
    ध्रुव HVX_Vector one  = hvx_vec_splat_f16((_Float16) 1.0f);
    HVX_VectorPred q = Q6_Q_vcmp_gt_VhfVhf(v, zero);
    प्रत्यागम Q6_V_vmux_QVV(q, one, zero);
}

स्थिर अन्तःस्थ HVX_VectorPred hvx_vec_is_nan_f16(HVX_Vector v) {
    ध्रुव HVX_Vector vnan_exp  = Q6_Vh_vsplat_R(0x7C00);
    ध्रुव HVX_Vector vnan_frac = Q6_Vh_vsplat_R(0x7FFF);

    // get pred of which are NaN, i.e., exponent bits all 1s and fraction bits non 0s
    HVX_VectorPred p_exp  = Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_exp), vnan_exp);
    HVX_VectorPred p_frac = Q6_Q_not_Q(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, vnan_frac), vnan_exp));
    प्रत्यागम Q6_Q_and_QQ(p_exp, p_frac);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_f32_to_f16_shuff(HVX_Vector v0, HVX_Vector v1) {
#यदि __HVX_ARCH__ >= 81
    HVX_Vector q0 = Q6_Vqf32_equals_Vsf(v0);
    HVX_Vector q1 = Q6_Vqf32_equals_Vsf(v1);
#अन्यथा
    ध्रुव HVX_Vector zero = Q6_V_vzero();
    HVX_Vector q0 = Q6_Vqf32_vadd_VsfVsf(v0, zero);
    HVX_Vector q1 = Q6_Vqf32_vadd_VsfVsf(v1, zero);
#endif
    प्रत्यागम Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_f32_to_f16(HVX_Vector v0, HVX_Vector v1) {
    प्रत्यागम Q6_Vh_vdeal_Vh(hvx_vec_f32_to_f16_shuff(v0, v1));
}

#यदि __HVX_ARCH__ >= 79
स्थिर अन्तःस्थ HVX_VectorPair hvx_vec_f16_to_f32_shuff(HVX_Vector v) {
    ध्रुव HVX_Vector one = hvx_vec_splat_f16(1.0);
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(v, one);
    प्रत्यागम Q6_W_vcombine_VV(Q6_V_hi_W(p), Q6_V_lo_W(p));
}
स्थिर अन्तःस्थ HVX_VectorPair hvx_vec_f16_to_f32(HVX_Vector v) {
    ध्रुव HVX_Vector one = hvx_vec_splat_f16(1.0);
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    प्रत्यागम Q6_W_vcombine_VV(Q6_V_hi_W(p), Q6_V_lo_W(p));
}
#अन्यथा
स्थिर अन्तःस्थ HVX_VectorPair hvx_vec_f16_to_f32_shuff(HVX_Vector v) {
    ध्रुव HVX_Vector one = hvx_vec_splat_f16(1.0);
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v, one);
    प्रत्यागम Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)));
}
स्थिर अन्तःस्थ HVX_VectorPair hvx_vec_f16_to_f32(HVX_Vector v) {
    ध्रुव HVX_Vector one = hvx_vec_splat_f16(1.0);
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
    प्रत्यागम Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)));
}
#endif

स्थिर अन्तःस्थ HVX_Vector hvx_vec_i16_from_hf_rnd_sat(HVX_Vector vin) {
    // This looks complicated.
    // Ideally should just be Q6_Vh_equals_Vhf(vin)
    // but that instruction does not do proper rounding.

    // convert to qf32, multiplying by 1.0 in the process.
    HVX_VectorPair v32 = Q6_Wqf32_vmpy_VhfVhf(vin, Q6_Vh_vsplat_R(0x3C00));

    // 'in-range' values are +/32752.
    // add 192K to it, convert to sf
    HVX_Vector v192K = Q6_V_vsplat_R(0x48400000);
    HVX_Vector vsf_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(v32), v192K));
    HVX_Vector vsf_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(v32), v192K));

    // for in-range cases, result is {163858... 229360} so the exponent is always 144.
    // if we extract bits 21..0 as a signed quantity, and round 6 bits off, that will be the answer.
    // Start by <<10 to get the final 'sign' bit in bit 15...
    vsf_0 = Q6_Vw_vasl_VwR(vsf_0, 10);
    vsf_1 = Q6_Vw_vasl_VwR(vsf_1, 10);

    // now round down to 16
    प्रत्यागम Q6_Vh_vround_VwVw_sat(vsf_1, vsf_0);
}

#यदि __HVX_ARCH__ < 79

स्थिर अन्तःस्थ HVX_VectorPair hvx_vec_mpyacc_f32_f16(HVX_VectorPair acc, HVX_Vector x, HVX_Vector y)
{
    HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(x, y);
    HVX_Vector a0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(m), Q6_V_lo_W(acc)));
    HVX_Vector a1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(m), Q6_V_hi_W(acc)));
    प्रत्यागम Q6_W_vcombine_VV(a1, a0);
}

#अन्यथा

स्थिर अन्तःस्थ HVX_VectorPair hvx_vec_mpyacc_f32_f16(HVX_VectorPair acc, HVX_Vector x, HVX_Vector y)
{
    प्रत्यागम Q6_Wsf_vmpyacc_WsfVhfVhf(acc, x, y);
}

#endif

#यदि __HVX_ARCH__ < 79

स्थिर अन्तःस्थ HVX_Vector hvx_vec_add_f16_f16(HVX_Vector a, HVX_Vector b)
{
    ध्रुव HVX_Vector negone = Q6_Vh_vsplat_R(0xBC00); // -1.0 in IEEE FP16
    ध्रुव HVX_Vector one    = Q6_Vh_vsplat_R(0x3C00); //  1.0 in IEEE FP16
    HVX_VectorPair a_p = Q6_Wqf32_vmpy_VhfVhf(a, one);
    HVX_VectorPair b_p = Q6_Wqf32_vmpy_VhfVhf(b, negone);
    HVX_Vector a0 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(a_p), Q6_V_lo_W(b_p));
    HVX_Vector a1 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(a_p), Q6_V_hi_W(b_p));
    प्रत्यागम Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1, a0));
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_sub_f16_f16(HVX_Vector a, HVX_Vector b)
{
    ध्रुव HVX_Vector negone = Q6_Vh_vsplat_R(0xBC00); // -1.0 in IEEE FP16
    ध्रुव HVX_Vector one    = Q6_Vh_vsplat_R(0x3C00); //  1.0 in IEEE FP16
    HVX_VectorPair a_p = Q6_Wqf32_vmpy_VhfVhf(a, one);
    HVX_VectorPair b_p = Q6_Wqf32_vmpy_VhfVhf(b, negone);
    HVX_Vector a0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(a_p), Q6_V_lo_W(b_p));
    HVX_Vector a1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(a_p), Q6_V_hi_W(b_p));
    प्रत्यागम Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1, a0));
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_mul_f16_f16(HVX_Vector a, HVX_Vector b)
{
    प्रत्यागम Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_add_f32_f32(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_sub_f32_f32(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_mul_f32_f32(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

#अन्यथा

स्थिर अन्तःस्थ HVX_Vector hvx_vec_add_f16_f16(HVX_Vector a, HVX_Vector b)
{
    प्रत्यागम Q6_Vhf_vadd_VhfVhf(a, b);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_sub_f16_f16(HVX_Vector a, HVX_Vector b)
{
    प्रत्यागम Q6_Vhf_vsub_VhfVhf(a, b);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_mul_f16_f16(HVX_Vector a, HVX_Vector b)
{
    प्रत्यागम Q6_Vhf_vmpy_VhfVhf(a, b);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_add_f32_f32(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_vadd_VsfVsf(a, b);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_sub_f32_f32(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_vsub_VsfVsf(a, b);
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_mul_f32_f32(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_vmpy_VsfVsf(a, b);
}

#endif // __HVX_ARCH__ < 79

स्थिर अन्तःस्थ HVX_Vector hvx_vec_load_act_tile(ध्रुव uint8_t * y_q, uint32_t kt, HVX_Vector * v_act_all) {
    यदि (kt % 4 == 0) {
        *v_act_all = hvx_vmem(y_q + kt * 32);
        प्रत्यागम *v_act_all;
    } अन्यथा यदि (kt % 4 == 1) {
        प्रत्यागम Q6_V_vror_VR(*v_act_all, 32);
    } अन्यथा यदि (kt % 4 == 2) {
        प्रत्यागम Q6_V_vror_VR(*v_act_all, 64);
    } अन्यथा {
        प्रत्यागम Q6_V_vror_VR(*v_act_all, 96);
    }
}

// Full tile, in place. tmp: 32 vectors scratch.
// Use when result must stay in m. Slowest (spills).
स्थिर अन्तःस्थ शून्य hvx_transpose_32x32_w_shuff_inplace(HVX_Vector * restrict m, HVX_Vector * restrict tmp) {
    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    प्रति (पूर्णाङ्क b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    प्रति (पूर्णाङ्क b = 0; b < 32; b += 8) {
        प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    प्रति (पूर्णाङ्क b = 0; b < 32; b += 16) {
        प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    प्रति (पूर्णाङ्क i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// Rows [0, nrows) of the transpose of src[0, ncols) to dst. src is clobbered, must not alias dst.
// Use when not in VTCM or for full tiles.
स्थिर अन्तःस्थ शून्य hvx_transpose_32x32_w_shuff(uint8_t * restrict dst, uint32_t dst_stride, HVX_Vector * restrict src,
                                               uint32_t nrows, uint32_t ncols) {
    प्रति (uint32_t g = 0; g < ncols; g += 8) {
        HVX_Vector * s = src + g;

        HVX_VectorPair a0 = Q6_W_vshuff_VVR(s[1], s[0], -4);
        HVX_VectorPair a1 = Q6_W_vshuff_VVR(s[3], s[2], -4);
        HVX_VectorPair a2 = Q6_W_vshuff_VVR(s[5], s[4], -4);
        HVX_VectorPair a3 = Q6_W_vshuff_VVR(s[7], s[6], -4);

        HVX_VectorPair b0 = Q6_W_vshuff_VVR(Q6_V_lo_W(a1), Q6_V_lo_W(a0), -8);
        HVX_VectorPair b1 = Q6_W_vshuff_VVR(Q6_V_hi_W(a1), Q6_V_hi_W(a0), -8);
        HVX_VectorPair b2 = Q6_W_vshuff_VVR(Q6_V_lo_W(a3), Q6_V_lo_W(a2), -8);
        HVX_VectorPair b3 = Q6_W_vshuff_VVR(Q6_V_hi_W(a3), Q6_V_hi_W(a2), -8);

        HVX_VectorPair c0 = Q6_W_vshuff_VVR(Q6_V_lo_W(b2), Q6_V_lo_W(b0), -16);
        HVX_VectorPair c1 = Q6_W_vshuff_VVR(Q6_V_hi_W(b2), Q6_V_hi_W(b0), -16);
        HVX_VectorPair c2 = Q6_W_vshuff_VVR(Q6_V_lo_W(b3), Q6_V_lo_W(b1), -16);
        HVX_VectorPair c3 = Q6_W_vshuff_VVR(Q6_V_hi_W(b3), Q6_V_hi_W(b1), -16);

        s[0] = Q6_V_lo_W(c0); s[1] = Q6_V_hi_W(c0);
        s[2] = Q6_V_lo_W(c1); s[3] = Q6_V_hi_W(c1);
        s[4] = Q6_V_lo_W(c2); s[5] = Q6_V_hi_W(c2);
        s[6] = Q6_V_lo_W(c3); s[7] = Q6_V_hi_W(c3);
    }

    प्रति (uint32_t k = 0; 4 * k < nrows; k++) {
        HVX_VectorPair d0 = Q6_W_vshuff_VVR(src[k + 8],  src[k],      -32);
        HVX_VectorPair d1 = Q6_W_vshuff_VVR(src[k + 24], src[k + 16], -32);

        HVX_VectorPair e0 = Q6_W_vshuff_VVR(Q6_V_lo_W(d1), Q6_V_lo_W(d0), -64);
        HVX_VectorPair e1 = Q6_W_vshuff_VVR(Q6_V_hi_W(d1), Q6_V_hi_W(d0), -64);

        uint8_t * d = dst + 4 * k * dst_stride;
        *(HVX_Vector *) (d + 0 * dst_stride) = Q6_V_lo_W(e0);
        *(HVX_Vector *) (d + 1 * dst_stride) = Q6_V_hi_W(e0);
        *(HVX_Vector *) (d + 2 * dst_stride) = Q6_V_lo_W(e1);
        *(HVX_Vector *) (d + 3 * dst_stride) = Q6_V_hi_W(e1);
    }
}

// Specialized store that will follow outstanding scatters or gathers to make sure they complete
स्थिर अन्तःस्थ शून्य hvx_gather_sync(शून्य * vtcm_addr) {
    यंत्र चञ्चल("vmem(%0+#0):scatter_release\n"
                 "v0 = vmem(%0+#0)\n" :: "r"(vtcm_addr) : "v0", "memory");
}

// Lane k = k * stride. Offsets for the _gather transposes.
स्थिर अन्तःस्थ HVX_Vector hvx_vec_gather_offsets_w(uint32_t stride) {
    int32_t offsets[32] __attribute__((aligned(128)));
    प्रति (पूर्णाङ्क k = 0; k < 32; k++) {
        offsets[k] = k * stride;
    }
    प्रत्यागम *(HVX_Vector *) offsets;
}

स्थिर अन्तःस्थ HVX_Vector hvx_vec_gather_offsets_h(uint32_t stride) {
    int16_t offsets[64] __attribute__((aligned(128)));
    प्रति (पूर्णाङ्क k = 0; k < 64; k++) {
        offsets[k] = k * stride;
    }
    प्रत्यागम *(HVX_Vector *) offsets;
}

// Rows [0, nrows) of the transpose of ncols src vectors (src_stride apart) to dst.
// src, dst in VTCM. Async: hvx_gather_sync() before use.
// Use for partial tiles (small ncols). Fewest packets.
स्थिर अन्तःस्थ शून्य hvx_transpose_32x32_w_gather(uint8_t * restrict dst, uint32_t dst_stride, ध्रुव uint8_t * restrict src,
                                                uint32_t src_stride, HVX_Vector offsets, uint32_t nrows, uint32_t ncols) {
    ध्रुव uint32_t mu = ncols * src_stride;

    #pragma unroll(4)
    प्रति (uint32_t r = 0; r < nrows; r++) {
        Q6_vgather_ARMVw((HVX_Vector *) (dst + r * dst_stride), (माप_प्रकार) (src + r * परिमाण(int32_t)), mu, offsets);
    }
}

// 16-bit version of hvx_transpose_32x32_w_gather.
स्थिर अन्तःस्थ शून्य hvx_transpose_64x64_h_gather(uint8_t * restrict dst, uint32_t dst_stride, ध्रुव uint8_t * restrict src,
                                                uint32_t src_stride, HVX_Vector offsets, uint32_t nrows, uint32_t ncols) {
    ध्रुव uint32_t mu = ncols * src_stride;

    #pragma unroll(4)
    प्रति (uint32_t r = 0; r < nrows; r++) {
        Q6_vgather_ARMVh((HVX_Vector *) (dst + r * dst_stride), (माप_प्रकार) (src + r * परिमाण(int16_t)), mu, offsets);
    }
}

#endif /* HVX_BASE_H */
