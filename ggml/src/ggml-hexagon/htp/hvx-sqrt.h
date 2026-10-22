#यदि_अपरिभाषितम् HVX_SQRT_H
#परिभाषय HVX_SQRT_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव "hex-utils.h"

#अन्तर्भाव "hvx-base.h"

#परिभाषय RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#परिभाषय RSQRT_ONE_HALF     0x3f000000  // 0.5
#परिभाषय RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#यदि __HVX_ARCH__ < 79
#परिभाषय HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#अन्यथा
#परिभाषय HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#यदि_अन्तः

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_rsqrt_f32(HVX_सदिशम् अन्तर्_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_सदिशम् rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_सदिशम् onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_सदिशम् threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_सदिशम् x2, y, ypower2, अस्थायी;

    x2 = Q6_Vqf32_vmpy_VsfVsf(अन्तर्_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(अन्तर्_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    अस्थायी    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    अस्थायी    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(अस्थायी));
    अस्थायी    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(अस्थायी));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(अस्थायी, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    अस्थायी    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    अस्थायी    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(अस्थायी));
    अस्थायी    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, अस्थायी);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(अस्थायी, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    अस्थायी    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    अस्थायी    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(अस्थायी));
    अस्थायी    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, अस्थायी);

    प्रत्यागम Q6_Vsf_equals_Vqf32(अस्थायी);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#परिभाषय hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    कुरु {                                                                     \
        dst_type * सीमित vdst = (dst_type *) dst;                         \
        src_type * सीमित vsrc = (src_type *) src;                         \
                                                                             \
        ध्रुव uint32_t nvec = n / VLEN_FP32;                                 \
        ध्रुव uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        प्रति (; i < nvec; i++) {                                              \
            HVX_सदिशम् inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_सदिशम् sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        यदि (nloe) {                                                          \
            HVX_सदिशम् inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_सदिशम् sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((शून्य *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } यावत्(0)

स्थिर अन्तःस्थ शून्य hvx_sqrt_f32_aa(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    प्रतिज्ञा((अचिह्नित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिह्नित दीर्घ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_सदिशम्, HVX_सदिशम्, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f32_au(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    प्रतिज्ञा((अचिह्नित दीर्घ) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_सदिशम्, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f32_ua(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    प्रतिज्ञा((अचिह्नित दीर्घ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_सदिशम्, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f32_uu(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f32(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    यदि ((अचिह्नित दीर्घ) dst % 128 == 0) {
        यदि ((अचिह्नित दीर्घ) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, अङ्कः_elems);
        } अन्यथा {
            hvx_sqrt_f32_au(dst, src, अङ्कः_elems);
        }
    } अन्यथा {
        यदि ((अचिह्नित दीर्घ) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, अङ्कः_elems);
        } अन्यथा {
            hvx_sqrt_f32_uu(dst, src, अङ्कः_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#परिभाषय hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    कुरु {                                                                               \
        dst_type * सीमित vdst = (dst_type *) dst;                                   \
        src_type * सीमित vsrc = (src_type *) src;                                   \
                                                                                       \
        ध्रुव uint32_t nvec = n / VLEN_FP16;                                           \
        ध्रुव uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        प्रति (; i < nvec; i++) {                                                        \
            HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(vsrc[i]);                            \
            HVX_सदिशम् r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_सदिशम् r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);                                      \
        }                                                                              \
        यदि (nloe) {                                                                    \
            HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(vsrc[i]);                            \
            HVX_सदिशम् r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_सदिशम् r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_सदिशम् v = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);                                 \
            vec_store((शून्य *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } यावत्(0)

स्थिर अन्तःस्थ शून्य hvx_sqrt_f16_aa(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    प्रतिज्ञा((अचिह्नित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिह्नित दीर्घ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_सदिशम्, HVX_सदिशम्, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f16_au(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    प्रतिज्ञा((अचिह्नित दीर्घ) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_सदिशम्, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f16_ua(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    प्रतिज्ञा((अचिह्नित दीर्घ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_सदिशम्, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f16_uu(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqrt_f16(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    यदि ((अचिह्नित दीर्घ) dst % 128 == 0) {
        यदि ((अचिह्नित दीर्घ) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, अङ्कः_elems);
        } अन्यथा {
            hvx_sqrt_f16_au(dst, src, अङ्कः_elems);
        }
    } अन्यथा {
        यदि ((अचिह्नित दीर्घ) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, अङ्कः_elems);
        } अन्यथा {
            hvx_sqrt_f16_uu(dst, src, अङ्कः_elems);
        }
    }
}

#यदि_अन्तः /* HVX_SQRT_H */
