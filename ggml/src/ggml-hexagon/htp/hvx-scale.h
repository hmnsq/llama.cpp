#अगर_अघोषित HVX_SCALE_H
#घोषणा HVX_SCALE_H

#समावेश <assert.h>
#समावेश <stddef.h>
#समावेश <stdint.h>

#समावेश "hvx-base.h"

#घोषणा hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    करो {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        स्थिर uint32_t elem_आकार = माप(भग्न);                                    \
        स्थिर uint32_t epv = 128 / elem_आकार;                                        \
        स्थिर uint32_t nvec = n / epv;                                               \
        स्थिर uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        क्रम (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        अगर (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } जबतक(0)

अटल अंतरभूत व्योम hvx_scale_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale) {
    निश्चित((माप_प्रकार) dst % 128 == 0);
    निश्चित((माप_प्रकार) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_scale_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale) {
    निश्चित((माप_प्रकार) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_scale_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale) {
    निश्चित((माप_प्रकार) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_scale_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_scale_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale) {
    अगर (((माप_प्रकार) dst & 127) == 0) {
        अगर (((माप_प्रकार) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } अन्यथा {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } अन्यथा {
        अगर (((माप_प्रकार) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } अन्यथा {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#घोषणा hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                     \
    करो {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src_type * restrict vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                         \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                        \
                                                                                          \
        स्थिर uint32_t elem_आकार = माप(भग्न);                                         \
        स्थिर uint32_t epv = 128 / elem_आकार;                                             \
        स्थिर uint32_t nvec = n / epv;                                                    \
        स्थिर uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        क्रम (; i < nvec; ++i) {                                                           \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        अगर (nloe) {                                                                       \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } जबतक(0)

अटल अंतरभूत व्योम hvx_scale_offset_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    निश्चित((माप_प्रकार) dst % 128 == 0);
    निश्चित((माप_प्रकार) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_scale_offset_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    निश्चित((माप_प्रकार) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_scale_offset_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    निश्चित((माप_प्रकार) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_scale_offset_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_scale_offset_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    अगर (((माप_प्रकार) dst & 127) == 0) {
        अगर (((माप_प्रकार) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } अन्यथा {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } अन्यथा {
        अगर (((माप_प्रकार) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } अन्यथा {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#घोषणा hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)                                                \
    करो {                                                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                                                 \
        src_type * restrict vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                                                   \
                                                                                                                     \
        स्थिर uint32_t nvec = n / VLEN_FP16;                                                                         \
        स्थिर uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        क्रम (; i < nvec; ++i) {                                                                                      \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                                                    \
        }                                                                                                            \
        अगर (nloe) {                                                                                                  \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                                               \
            vec_store((व्योम *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } जबतक(0)

अटल अंतरभूत व्योम hvx_scale_offset_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    निश्चित((माप_प्रकार) dst % 128 == 0);
    निश्चित((माप_प्रकार) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_scale_offset_f16_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    निश्चित((माप_प्रकार) dst % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_scale_offset_f16_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    निश्चित((माप_प्रकार) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_scale_offset_f16_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_scale_offset_f16(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर पूर्णांक n, स्थिर भग्न scale, स्थिर भग्न offset) {
    अगर (((माप_प्रकार) dst & 127) == 0) {
        अगर (((माप_प्रकार) src & 127) == 0) {
            hvx_scale_offset_f16_aa(dst, src, n, scale, offset);
        } अन्यथा {
            hvx_scale_offset_f16_au(dst, src, n, scale, offset);
        }
    } अन्यथा {
        अगर (((माप_प्रकार) src & 127) == 0) {
            hvx_scale_offset_f16_ua(dst, src, n, scale, offset);
        } अन्यथा {
            hvx_scale_offset_f16_uu(dst, src, n, scale, offset);
        }
    }
}

#पूर्ण_अगर // HVX_SCALE_H
