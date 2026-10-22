#अगर_अघोषित HVX_ARITH_H
#घोषणा HVX_ARITH_H

#समावेश <assert.h>
#समावेश <stddef.h>
#समावेश <stdint.h>
#समावेश <math.h>

#समावेश "hvx-base.h"
#समावेश "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#घोषणा UNUSED(x) (व्योम)(x)

#घोषणा hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_आकार, vec_store, vec_op) \
    करो {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        स्थिर uint32_t epv  = 128 / (elem_आकार);                                          \
        स्थिर uint32_t nvec = n / epv;                                                    \
        स्थिर uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        क्रम (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        अगर (nloe) {                                                                       \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((व्योम *) &vdst[i], nloe * (elem_आकार), v);                          \
        }                                                                                 \
    } जबतक(0)

#अगर __HVX_ARCH__ < 79

#घोषणा HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#घोषणा HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#घोषणा HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#अन्यथा

#घोषणा HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#घोषणा HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#घोषणा HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#पूर्ण_अगर

#घोषणा HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#घोषणा HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#घोषणा HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#घोषणा DEFINE_HVX_द्विआधारी_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
अटल अंतरभूत व्योम OP_NAME##_aaa(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    निश्चित((uintptr_t) dst % 128 == 0);                                                                       \
    निश्चित((uintptr_t) src0 % 128 == 0);                                                                      \
    निश्चित((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, माप(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
अटल अंतरभूत व्योम OP_NAME##_aau(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    निश्चित((uintptr_t) dst % 128 == 0);                                                                       \
    निश्चित((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, माप(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
अटल अंतरभूत व्योम OP_NAME##_aua(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    निश्चित((uintptr_t) dst % 128 == 0);                                                                       \
    निश्चित((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, माप(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
अटल अंतरभूत व्योम OP_NAME##_auu(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    निश्चित((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, माप(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
अटल अंतरभूत व्योम OP_NAME##_uaa(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    निश्चित((uintptr_t) src0 % 128 == 0);                                                                      \
    निश्चित((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, माप(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
अटल अंतरभूत व्योम OP_NAME##_uau(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    निश्चित((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, माप(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
अटल अंतरभूत व्योम OP_NAME##_uua(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    निश्चित((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, माप(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
अटल अंतरभूत व्योम OP_NAME##_uuu(uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, माप(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, भग्न)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, भग्न)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, भग्न)

DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#घोषणा HVX_द्विआधारी_DISPATCHER(OP_NAME)                                                                                                       \
अटल अंतरभूत व्योम OP_NAME(uint8_t * restrict dst, स्थिर uint8_t * restrict src0, स्थिर uint8_t * restrict src1, स्थिर uint32_t अंक_elems) { \
    अगर (hex_is_aligned((व्योम *) dst, 128)) {                                                                                                 \
        अगर (hex_is_aligned((व्योम *) src0, 128)) {                                                                                            \
            अगर (hex_is_aligned((व्योम *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, अंक_elems);                                               \
            अन्यथा                                    OP_NAME##_aau(dst, src0, src1, अंक_elems);                                               \
        } अन्यथा {                                                                                                                             \
            अगर (hex_is_aligned((व्योम *) src1, 128)) OP_NAME##_aua(dst, src0, src1, अंक_elems);                                               \
            अन्यथा                                    OP_NAME##_auu(dst, src0, src1, अंक_elems);                                               \
        }                                                                                                                                    \
    } अन्यथा {                                                                                                                                 \
        अगर (hex_is_aligned((व्योम *) src0, 128)) {                                                                                            \
            अगर (hex_is_aligned((व्योम *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, अंक_elems);                                               \
            अन्यथा                                    OP_NAME##_uau(dst, src0, src1, अंक_elems);                                               \
        } अन्यथा {                                                                                                                             \
            अगर (hex_is_aligned((व्योम *) src1, 128)) OP_NAME##_uua(dst, src0, src1, अंक_elems);                                               \
            अन्यथा                                    OP_NAME##_uuu(dst, src0, src1, अंक_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_द्विआधारी_DISPATCHER(hvx_add_f32)
HVX_द्विआधारी_DISPATCHER(hvx_sub_f32)
HVX_द्विआधारी_DISPATCHER(hvx_mul_f32)

HVX_द्विआधारी_DISPATCHER(hvx_add_f16)
HVX_द्विआधारी_DISPATCHER(hvx_sub_f16)
HVX_द्विआधारी_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
अटल अंतरभूत व्योम hvx_mul_mul_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src0, स्थिर uint8_t * restrict src1, स्थिर uint8_t * restrict src2, स्थिर uint32_t अंक_elems) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src0 % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src1 % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    स्थिर uint32_t elem_आकार = माप(भग्न);
    स्थिर uint32_t epv  = 128 / elem_आकार;
    स्थिर uint32_t nvec = अंक_elems / epv;
    स्थिर uint32_t nloe = अंक_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    अगर (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((व्योम *) &vdst[i], nloe * elem_आकार, v2);
    }
}

अटल अंतरभूत व्योम hvx_mul_mul_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src0, स्थिर uint8_t * restrict src1, स्थिर uint8_t * restrict src2, स्थिर uint32_t अंक_elems) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src0 % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src1 % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    स्थिर uint32_t elem_आकार = माप(_Float16);
    स्थिर uint32_t epv  = 128 / elem_आकार;
    स्थिर uint32_t nvec = अंक_elems / epv;
    स्थिर uint32_t nloe = अंक_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    अगर (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((व्योम *) &vdst[i], nloe * elem_आकार, v2);
    }
}

// Scalar Operations

#घोषणा hvx_scalar_loop_body(dst_type, src_type, elem_आकार, vec_store, scalar_op_macro) \
    करो {                                                                                \
        dst_type * restrict vdst = (dst_type *) dst;                                    \
        src_type * restrict vsrc = (src_type *) src;                                    \
                                                                                        \
        स्थिर uint32_t epv  = 128 / (elem_आकार);                                        \
        स्थिर uint32_t nvec = n / epv;                                                  \
        स्थिर uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        क्रम (; i < nvec; i++) {                                                         \
            HVX_Vector v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        अगर (nloe) {                                                                     \
            HVX_Vector v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((व्योम *) &vdst[i], nloe * (elem_आकार), v);                        \
        }                                                                               \
    } जबतक(0)

#घोषणा HVX_OP_ADD_SCALAR_F32(v)                                   \
    ({                                                             \
        स्थिर HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#घोषणा HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#घोषणा HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#घोषणा HVX_OP_ADD_SCALAR_F16(v)                                   \
    ({                                                             \
        स्थिर HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#घोषणा HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#घोषणा HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#घोषणा DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
अटल अंतरभूत व्योम OP_NAME##_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर ELEM_TYPE val, uint32_t n) { \
    स्थिर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    स्थिर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    निश्चित((uintptr_t) dst % 128 == 0);                                                                                  \
    निश्चित((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, माप(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
अटल अंतरभूत व्योम OP_NAME##_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर ELEM_TYPE val, uint32_t n) { \
    स्थिर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    स्थिर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    निश्चित((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, माप(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
अटल अंतरभूत व्योम OP_NAME##_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर ELEM_TYPE val, uint32_t n) { \
    स्थिर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    स्थिर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    निश्चित((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, माप(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
अटल अंतरभूत व्योम OP_NAME##_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर ELEM_TYPE val, uint32_t n) { \
    स्थिर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    स्थिर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, माप(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, भग्न)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, भग्न)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, भग्न)

DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#घोषणा HVX_द्विआधारी_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
अटल अंतरभूत व्योम OP_NAME(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर ELEM_TYPE val, स्थिर uint32_t अंक_elems) { \
    अगर (hex_is_aligned((व्योम *) dst, 128) && hex_is_aligned((व्योम *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, अंक_elems);                                                                                   \
    } अन्यथा अगर (hex_is_aligned((व्योम *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, अंक_elems);                                                                                   \
    } अन्यथा अगर (hex_is_aligned((व्योम *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, अंक_elems);                                                                                   \
    } अन्यथा {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, अंक_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_add_scalar_f32, भग्न)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_sub_scalar_f32, भग्न)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_mul_scalar_f32, भग्न)

HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#घोषणा HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

अटल अंतरभूत व्योम hvx_min_scalar_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, माप(भग्न), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

अटल अंतरभूत व्योम hvx_min_scalar_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, माप(भग्न), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

अटल अंतरभूत व्योम hvx_min_scalar_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, माप(भग्न), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

अटल अंतरभूत व्योम hvx_min_scalar_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, माप(भग्न), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

अटल अंतरभूत व्योम hvx_min_scalar_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128) && hex_is_aligned((व्योम *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, अंक_elems);
    } अन्यथा {
        hvx_min_scalar_f32_uu(dst, src, val, अंक_elems);
    }
}

// MAX Scalar variants

#घोषणा HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

अटल अंतरभूत व्योम hvx_max_scalar_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, माप(भग्न), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

अटल अंतरभूत व्योम hvx_max_scalar_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, माप(भग्न), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

अटल अंतरभूत व्योम hvx_max_scalar_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, माप(भग्न), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

अटल अंतरभूत व्योम hvx_max_scalar_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, माप(भग्न), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

अटल अंतरभूत व्योम hvx_max_scalar_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128) && hex_is_aligned((व्योम *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, अंक_elems);
    } अन्यथा {
        hvx_max_scalar_f32_uu(dst, src, val, अंक_elems);
    }
}

#घोषणा HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

अटल अंतरभूत व्योम hvx_min_scalar_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, माप(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_min_scalar_f16_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, माप(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_min_scalar_f16_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, माप(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_min_scalar_f16_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 val, uint32_t n) {
    स्थिर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, माप(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_min_scalar_f16(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 val, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128) && hex_is_aligned((व्योम *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, अंक_elems);
    } अन्यथा {
        hvx_min_scalar_f16_uu(dst, src, val, अंक_elems);
    }
}

// CLAMP Scalar variants

#घोषणा HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_VectorPred pred_cap_दायाँ = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_बायाँ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_दायाँ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_बायाँ, min_vec, tmp);                      \
    })

अटल अंतरभूत व्योम hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न min, स्थिर भग्न max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, माप(भग्न), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न min, स्थिर भग्न max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, माप(भग्न), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न min, स्थिर भग्न max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, माप(भग्न), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न min, स्थिर भग्न max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, माप(भग्न), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न min, स्थिर भग्न max, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128) && hex_is_aligned((व्योम *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, अंक_elems);
    } अन्यथा {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, अंक_elems);
    }
}

#घोषणा HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_VectorPred pred_cap_दायाँ = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_VectorPred pred_cap_बायाँ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_दायाँ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_बायाँ, min_vec, tmp);                      \
    })

अटल अंतरभूत व्योम hvx_clamp_scalar_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 min, स्थिर _Float16 max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, माप(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f16_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 min, स्थिर _Float16 max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, माप(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f16_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 min, स्थिर _Float16 max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, माप(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f16_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 min, स्थिर _Float16 max, uint32_t n) {
    स्थिर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    स्थिर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, माप(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

अटल अंतरभूत व्योम hvx_clamp_scalar_f16(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 min, स्थिर _Float16 max, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128) && hex_is_aligned((व्योम *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, अंक_elems);
    } अन्यथा {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, अंक_elems);
    }
}

#घोषणा HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_VectorPred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_Vector     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

अटल अंतरभूत व्योम hvx_leaky_relu_scalar_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न ns, uint32_t n) {
    स्थिर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    स्थिर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, माप(भग्न), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

अटल अंतरभूत व्योम hvx_leaky_relu_scalar_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न ns, uint32_t n) {
    स्थिर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    स्थिर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, माप(भग्न), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

अटल अंतरभूत व्योम hvx_leaky_relu_scalar_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न ns, uint32_t n) {
    स्थिर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    स्थिर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, माप(भग्न), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

अटल अंतरभूत व्योम hvx_leaky_relu_scalar_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न ns, uint32_t n) {
    स्थिर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    स्थिर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, माप(भग्न), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

अटल अंतरभूत व्योम hvx_leaky_relu_scalar_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न ns, स्थिर पूर्णांक अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128) && hex_is_aligned((व्योम *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, अंक_elems);
    } अन्यथा अगर (hex_is_aligned((व्योम *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, अंक_elems);
    } अन्यथा {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, अंक_elems);
    }
}

//
// Abs
//

अटल अंतरभूत व्योम hvx_abs_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    स्थिर uint32_t elem_आकार = माप(भग्न);
    स्थिर uint32_t epv       = 128 / elem_आकार;
    स्थिर uint32_t nvec      = n / epv;
    स्थिर uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    अगर (nloe) {
        HVX_Vector v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((व्योम *) &vdst[i], nloe * elem_आकार, v);
    }
}

#घोषणा hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    करो {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        स्थिर uint32_t elem_आकार = माप(_Float16);                       \
        स्थिर uint32_t epv  = 128 / elem_आकार;                             \
        स्थिर uint32_t nvec = n / epv;                                     \
        स्थिर uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        क्रम (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        अगर (nloe) {                                                        \
            HVX_Vector v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, v);             \
        }                                                                  \
    } जबतक(0)

अटल अंतरभूत व्योम hvx_abs_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_abs_f16_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_abs_f16_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_abs_f16_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_abs_f16(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर uint32_t अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128)) {
        अगर (hex_is_aligned((व्योम *) src, 128)) {
            hvx_abs_f16_aa(dst, src, अंक_elems);
        } अन्यथा {
            hvx_abs_f16_au(dst, src, अंक_elems);
        }
    } अन्यथा {
        अगर (hex_is_aligned((व्योम *) src, 128)) {
            hvx_abs_f16_ua(dst, src, अंक_elems);
        } अन्यथा {
            hvx_abs_f16_uu(dst, src, अंक_elems);
        }
    }
}

//
// Step
//

अटल अंतरभूत व्योम hvx_step_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    स्थिर uint32_t elem_आकार = माप(भग्न);
    स्थिर uint32_t epv       = 128 / elem_आकार;
    स्थिर uint32_t nvec      = n / epv;
    स्थिर uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f32(vsrc[i]);
    }
    अगर (nloe) {
        HVX_Vector v = hvx_vec_step_f32(vsrc[i]);
        hvx_vec_store_a((व्योम *) &vdst[i], nloe * elem_आकार, v);
    }
}

अटल अंतरभूत व्योम hvx_step_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    स्थिर uint32_t elem_आकार = माप(_Float16);
    स्थिर uint32_t epv       = 128 / elem_आकार;
    स्थिर uint32_t nvec      = n / epv;
    स्थिर uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f16(vsrc[i]);
    }
    अगर (nloe) {
        HVX_Vector v = hvx_vec_step_f16(vsrc[i]);
        hvx_vec_store_a((व्योम *) &vdst[i], nloe * elem_आकार, v);
    }
}

//
// Square
//

#घोषणा hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    करो {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        स्थिर uint32_t elem_आकार = माप(भग्न);                          \
        स्थिर uint32_t epv  = 128 / elem_आकार;                             \
        स्थिर uint32_t nvec = n / epv;                                     \
        स्थिर uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        क्रम (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        अगर (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, v);             \
        }                                                                  \
    } जबतक(0)

अटल अंतरभूत व्योम hvx_sqr_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_sqr_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_sqr_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_sqr_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_sqr_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर uint32_t अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128)) {
        अगर (hex_is_aligned((व्योम *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, अंक_elems);
        } अन्यथा {
            hvx_sqr_f32_au(dst, src, अंक_elems);
        }
    } अन्यथा {
        अगर (hex_is_aligned((व्योम *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, अंक_elems);
        } अन्यथा {
            hvx_sqr_f32_uu(dst, src, अंक_elems);
        }
    }
}

#घोषणा hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    करो {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        स्थिर uint32_t elem_आकार = माप(_Float16);                       \
        स्थिर uint32_t epv  = 128 / elem_आकार;                             \
        स्थिर uint32_t nvec = n / epv;                                     \
        स्थिर uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        क्रम (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        अगर (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, v);             \
        }                                                                  \
    } जबतक(0)

अटल अंतरभूत व्योम hvx_sqr_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_sqr_f16_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_sqr_f16_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_sqr_f16_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_sqr_f16(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर uint32_t अंक_elems) {
    अगर (hex_is_aligned((व्योम *) dst, 128)) {
        अगर (hex_is_aligned((व्योम *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, अंक_elems);
        } अन्यथा {
            hvx_sqr_f16_au(dst, src, अंक_elems);
        }
    } अन्यथा {
        अगर (hex_is_aligned((व्योम *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, अंक_elems);
        } अन्यथा {
            hvx_sqr_f16_uu(dst, src, अंक_elems);
        }
    }
}

#अघोषित HVX_OP_ADD_F32
#अघोषित HVX_OP_SUB_F32
#अघोषित HVX_OP_MUL_F32
#अघोषित HVX_OP_ADD_F16
#अघोषित HVX_OP_SUB_F16
#अघोषित HVX_OP_MUL_F16
#अघोषित hvx_arith_loop_body
#अघोषित HVX_OP_ADD_SCALAR_F32
#अघोषित HVX_OP_SUB_SCALAR_F32
#अघोषित HVX_OP_MUL_SCALAR_F32
#अघोषित HVX_OP_ADD_SCALAR_F16
#अघोषित HVX_OP_SUB_SCALAR_F16
#अघोषित HVX_OP_MUL_SCALAR_F16
#अघोषित hvx_scalar_loop_body
#अघोषित HVX_OP_MIN_SCALAR
#अघोषित HVX_OP_MAX_SCALAR
#अघोषित HVX_OP_MIN_SCALAR_F16
#अघोषित HVX_OP_CLAMP_SCALAR
#अघोषित HVX_OP_CLAMP_SCALAR_F16
#अघोषित HVX_OP_LEAKY_RELU_SCALAR
#अघोषित DEFINE_HVX_द्विआधारी_OP_VARIANTS
#अघोषित DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS
#अघोषित HVX_द्विआधारी_DISPATCHER
#अघोषित UNUSED

#पूर्ण_अगर // HVX_ARITH_H
