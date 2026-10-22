#यदि_अपरिभाषितम् HVX_SCALE_H
#परिभाषय HVX_SCALE_H

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव "hvx-base.h"

#परिभाषय hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    कुरु {                                                                             \
        dst_type * सीमित vdst = (dst_type *) dst;                                 \
        src_type * सीमित vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_सदिशम् vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ध्रुव uint32_t elem_आकार = परिमाण(भिन्न);                                    \
        ध्रुव uint32_t epv = 128 / elem_आकार;                                        \
        ध्रुव uint32_t nvec = n / epv;                                               \
        ध्रुव uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        प्रति (; i < nvec; ++i) {                                                      \
            HVX_सदिशम् v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        यदि (nloe) {                                                                  \
            HVX_सदिशम् v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((शून्य *) &vdst[i], nloe * elem_आकार, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } यावत्(0)

स्थिर अन्तःस्थ शून्य hvx_scale_f32_aa(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale) {
    प्रतिज्ञा((size_t) dst % 128 == 0);
    प्रतिज्ञा((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_सदिशम्, HVX_सदिशम्, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_scale_f32_au(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale) {
    प्रतिज्ञा((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_सदिशम्, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_scale_f32_ua(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale) {
    प्रतिज्ञा((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_सदिशम्, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_scale_f32_uu(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_scale_f32(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale) {
    यदि (((size_t) dst & 127) == 0) {
        यदि (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } अन्यथा {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } अन्यथा {
        यदि (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } अन्यथा {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#परिभाषय hvx_scale_विस्थापनम्_f32_loop_body(dst_type, src_type, vec_store)                     \
    कुरु {                                                                                  \
        dst_type * सीमित vdst = (dst_type *) dst;                                      \
        src_type * सीमित vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_सदिशम् vs = hvx_vec_splat_f32(scale);                                         \
        HVX_सदिशम् vo = hvx_vec_splat_f32(विस्थापनम्);                                        \
                                                                                          \
        ध्रुव uint32_t elem_आकार = परिमाण(भिन्न);                                         \
        ध्रुव uint32_t epv = 128 / elem_आकार;                                             \
        ध्रुव uint32_t nvec = n / epv;                                                    \
        ध्रुव uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        प्रति (; i < nvec; ++i) {                                                           \
            HVX_सदिशम् v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        यदि (nloe) {                                                                       \
            HVX_सदिशम् v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((शून्य *) &vdst[i], nloe * elem_आकार, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } यावत्(0)

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f32_aa(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    प्रतिज्ञा((size_t) dst % 128 == 0);
    प्रतिज्ञा((size_t) src % 128 == 0);
    hvx_scale_विस्थापनम्_f32_loop_body(HVX_सदिशम्, HVX_सदिशम्, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f32_au(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    प्रतिज्ञा((size_t) dst % 128 == 0);
    hvx_scale_विस्थापनम्_f32_loop_body(HVX_सदिशम्, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f32_ua(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    प्रतिज्ञा((size_t) src % 128 == 0);
    hvx_scale_विस्थापनम्_f32_loop_body(HVX_UVector, HVX_सदिशम्, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f32_uu(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    hvx_scale_विस्थापनम्_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f32(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    यदि (((size_t) dst & 127) == 0) {
        यदि (((size_t) src & 127) == 0) {
            hvx_scale_विस्थापनम्_f32_aa(dst, src, n, scale, विस्थापनम्);
        } अन्यथा {
            hvx_scale_विस्थापनम्_f32_au(dst, src, n, scale, विस्थापनम्);
        }
    } अन्यथा {
        यदि (((size_t) src & 127) == 0) {
            hvx_scale_विस्थापनम्_f32_ua(dst, src, n, scale, विस्थापनम्);
        } अन्यथा {
            hvx_scale_विस्थापनम्_f32_uu(dst, src, n, scale, विस्थापनम्);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#परिभाषय hvx_scale_विस्थापनम्_f16_loop_body(dst_type, src_type, vec_store)                                                \
    कुरु {                                                                                                             \
        dst_type * सीमित vdst = (dst_type *) dst;                                                                 \
        src_type * सीमित vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_सदिशम् vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_सदिशम् vo = hvx_vec_splat_f32(विस्थापनम्);                                                                   \
                                                                                                                     \
        ध्रुव uint32_t nvec = n / VLEN_FP16;                                                                         \
        ध्रुव uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        प्रति (; i < nvec; ++i) {                                                                                      \
            HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(vsrc[i]);                                                          \
            HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);                                                                    \
        }                                                                                                            \
        यदि (nloe) {                                                                                                  \
            HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(vsrc[i]);                                                          \
            HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_सदिशम् v = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);                                                               \
            vec_store((शून्य *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } यावत्(0)

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f16_aa(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    प्रतिज्ञा((size_t) dst % 128 == 0);
    प्रतिज्ञा((size_t) src % 128 == 0);
    hvx_scale_विस्थापनम्_f16_loop_body(HVX_सदिशम्, HVX_सदिशम्, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f16_au(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    प्रतिज्ञा((size_t) dst % 128 == 0);
    hvx_scale_विस्थापनम्_f16_loop_body(HVX_सदिशम्, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f16_ua(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    प्रतिज्ञा((size_t) src % 128 == 0);
    hvx_scale_विस्थापनम्_f16_loop_body(HVX_UVector, HVX_सदिशम्, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f16_uu(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    hvx_scale_विस्थापनम्_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_scale_विस्थापनम्_f16(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव पूर्णाङ्क n, ध्रुव भिन्न scale, ध्रुव भिन्न विस्थापनम्) {
    यदि (((size_t) dst & 127) == 0) {
        यदि (((size_t) src & 127) == 0) {
            hvx_scale_विस्थापनम्_f16_aa(dst, src, n, scale, विस्थापनम्);
        } अन्यथा {
            hvx_scale_विस्थापनम्_f16_au(dst, src, n, scale, विस्थापनम्);
        }
    } अन्यथा {
        यदि (((size_t) src & 127) == 0) {
            hvx_scale_विस्थापनम्_f16_ua(dst, src, n, scale, विस्थापनम्);
        } अन्यथा {
            hvx_scale_विस्थापनम्_f16_uu(dst, src, n, scale, विस्थापनम्);
        }
    }
}

#यदि_अन्तः // HVX_SCALE_H
