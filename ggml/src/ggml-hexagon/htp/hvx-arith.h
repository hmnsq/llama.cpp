#ifndef HVX_ARITH_H
#define HVX_ARITH_H

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <math.h>

#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#define UNUSED(x) (शून्य)(x)

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_size, vec_store, vec_op) \
    गर {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        अचर uint32_t epv  = 128 / (elem_size);                                          \
        अचर uint32_t nvec = n / epv;                                                    \
        अचर uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        लागि (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        यदि (nloe) {                                                                       \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((शून्य *) &vdst[i], nloe * (elem_size), v);                          \
        }                                                                                 \
    } जबसम्म(0)

#यदि __HVX_ARCH__ < 79

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#अन्यथा

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#endif

#define HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#define HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#define HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_द्विआधारी_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
स्थिर अन्तःस्थ शून्य OP_NAME##_aaa(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    प्रतिज्ञा((uintptr_t) dst % 128 == 0);                                                                       \
    प्रतिज्ञा((uintptr_t) src0 % 128 == 0);                                                                      \
    प्रतिज्ञा((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, परिमाण(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
स्थिर अन्तःस्थ शून्य OP_NAME##_aau(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    प्रतिज्ञा((uintptr_t) dst % 128 == 0);                                                                       \
    प्रतिज्ञा((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, परिमाण(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
स्थिर अन्तःस्थ शून्य OP_NAME##_aua(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    प्रतिज्ञा((uintptr_t) dst % 128 == 0);                                                                       \
    प्रतिज्ञा((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, परिमाण(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
स्थिर अन्तःस्थ शून्य OP_NAME##_auu(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    प्रतिज्ञा((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, परिमाण(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
स्थिर अन्तःस्थ शून्य OP_NAME##_uaa(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    प्रतिज्ञा((uintptr_t) src0 % 128 == 0);                                                                      \
    प्रतिज्ञा((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, परिमाण(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
स्थिर अन्तःस्थ शून्य OP_NAME##_uau(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    प्रतिज्ञा((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, परिमाण(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
स्थिर अन्तःस्थ शून्य OP_NAME##_uua(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    प्रतिज्ञा((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, परिमाण(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
स्थिर अन्तःस्थ शून्य OP_NAME##_uuu(uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, परिमाण(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, भिन्न)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, भिन्न)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, भिन्न)

DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_द्विआधारी_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#define HVX_द्विआधारी_DISPATCHER(OP_NAME)                                                                                                       \
स्थिर अन्तःस्थ शून्य OP_NAME(uint8_t * restrict dst, अचर uint8_t * restrict src0, अचर uint8_t * restrict src1, अचर uint32_t अङ्क_elems) { \
    यदि (hex_is_aligned((शून्य *) dst, 128)) {                                                                                                 \
        यदि (hex_is_aligned((शून्य *) src0, 128)) {                                                                                            \
            यदि (hex_is_aligned((शून्य *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, अङ्क_elems);                                               \
            अन्यथा                                    OP_NAME##_aau(dst, src0, src1, अङ्क_elems);                                               \
        } अन्यथा {                                                                                                                             \
            यदि (hex_is_aligned((शून्य *) src1, 128)) OP_NAME##_aua(dst, src0, src1, अङ्क_elems);                                               \
            अन्यथा                                    OP_NAME##_auu(dst, src0, src1, अङ्क_elems);                                               \
        }                                                                                                                                    \
    } अन्यथा {                                                                                                                                 \
        यदि (hex_is_aligned((शून्य *) src0, 128)) {                                                                                            \
            यदि (hex_is_aligned((शून्य *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, अङ्क_elems);                                               \
            अन्यथा                                    OP_NAME##_uau(dst, src0, src1, अङ्क_elems);                                               \
        } अन्यथा {                                                                                                                             \
            यदि (hex_is_aligned((शून्य *) src1, 128)) OP_NAME##_uua(dst, src0, src1, अङ्क_elems);                                               \
            अन्यथा                                    OP_NAME##_uuu(dst, src0, src1, अङ्क_elems);                                               \
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
स्थिर अन्तःस्थ शून्य hvx_mul_mul_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src0, अचर uint8_t * restrict src1, अचर uint8_t * restrict src2, अचर uint32_t अङ्क_elems) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src0 % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src1 % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    अचर uint32_t elem_size = परिमाण(भिन्न);
    अचर uint32_t epv  = 128 / elem_size;
    अचर uint32_t nvec = अङ्क_elems / epv;
    अचर uint32_t nloe = अङ्क_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    लागि (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    यदि (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((शून्य *) &vdst[i], nloe * elem_size, v2);
    }
}

स्थिर अन्तःस्थ शून्य hvx_mul_mul_f16_aa(uint8_t * restrict dst, अचर uint8_t * restrict src0, अचर uint8_t * restrict src1, अचर uint8_t * restrict src2, अचर uint32_t अङ्क_elems) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src0 % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src1 % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    अचर uint32_t elem_size = परिमाण(_Float16);
    अचर uint32_t epv  = 128 / elem_size;
    अचर uint32_t nvec = अङ्क_elems / epv;
    अचर uint32_t nloe = अङ्क_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    लागि (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    यदि (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((शून्य *) &vdst[i], nloe * elem_size, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, elem_size, vec_store, scalar_op_macro) \
    गर {                                                                                \
        dst_type * restrict vdst = (dst_type *) dst;                                    \
        src_type * restrict vsrc = (src_type *) src;                                    \
                                                                                        \
        अचर uint32_t epv  = 128 / (elem_size);                                        \
        अचर uint32_t nvec = n / epv;                                                  \
        अचर uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        लागि (; i < nvec; i++) {                                                         \
            HVX_Vector v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        यदि (nloe) {                                                                     \
            HVX_Vector v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((शून्य *) &vdst[i], nloe * (elem_size), v);                        \
        }                                                                               \
    } जबसम्म(0)

#define HVX_OP_ADD_SCALAR_F32(v)                                   \
    ({                                                             \
        अचर HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#define HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#define HVX_OP_ADD_SCALAR_F16(v)                                   \
    ({                                                             \
        अचर HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#define HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
स्थिर अन्तःस्थ शून्य OP_NAME##_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर ELEM_TYPE val, uint32_t n) { \
    अचर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    अचर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    प्रतिज्ञा((uintptr_t) dst % 128 == 0);                                                                                  \
    प्रतिज्ञा((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, परिमाण(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
स्थिर अन्तःस्थ शून्य OP_NAME##_au(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर ELEM_TYPE val, uint32_t n) { \
    अचर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    अचर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    प्रतिज्ञा((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, परिमाण(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
स्थिर अन्तःस्थ शून्य OP_NAME##_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर ELEM_TYPE val, uint32_t n) { \
    अचर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    अचर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    प्रतिज्ञा((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, परिमाण(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
स्थिर अन्तःस्थ शून्य OP_NAME##_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर ELEM_TYPE val, uint32_t n) { \
    अचर HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    अचर HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)अनन्त); UNUSED(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, परिमाण(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, भिन्न)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, भिन्न)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, भिन्न)

DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#define HVX_द्विआधारी_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
स्थिर अन्तःस्थ शून्य OP_NAME(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर ELEM_TYPE val, अचर uint32_t अङ्क_elems) { \
    यदि (hex_is_aligned((शून्य *) dst, 128) && hex_is_aligned((शून्य *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, अङ्क_elems);                                                                                   \
    } अन्यथा यदि (hex_is_aligned((शून्य *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, अङ्क_elems);                                                                                   \
    } अन्यथा यदि (hex_is_aligned((शून्य *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, अङ्क_elems);                                                                                   \
    } अन्यथा {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, अङ्क_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_add_scalar_f32, भिन्न)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_sub_scalar_f32, भिन्न)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_mul_scalar_f32, भिन्न)

HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_द्विआधारी_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f32_au(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f32_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f32_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f32(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, अचर पूर्णाङ्क अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128) && hex_is_aligned((शून्य *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, अङ्क_elems);
    } अन्यथा {
        hvx_min_scalar_f32_uu(dst, src, val, अङ्क_elems);
    }
}

// MAX Scalar variants

#define HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

स्थिर अन्तःस्थ शून्य hvx_max_scalar_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_max_scalar_f32_au(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_max_scalar_f32_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_max_scalar_f32_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_max_scalar_f32(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, अचर पूर्णाङ्क अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128) && hex_is_aligned((शून्य *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, अङ्क_elems);
    } अन्यथा {
        hvx_max_scalar_f32_uu(dst, src, val, अङ्क_elems);
    }
}

#define HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f16_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, परिमाण(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f16_au(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, परिमाण(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f16_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, परिमाण(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f16_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 val, uint32_t n) {
    अचर HVX_Vector val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, परिमाण(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_min_scalar_f16(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 val, अचर पूर्णाङ्क अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128) && hex_is_aligned((शून्य *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, अङ्क_elems);
    } अन्यथा {
        hvx_min_scalar_f16_uu(dst, src, val, अङ्क_elems);
    }
}

// CLAMP Scalar variants

#define HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_VectorPred pred_cap_दायाँ = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_बायाँ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_दायाँ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_बायाँ, min_vec, tmp);                      \
    })

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न min, अचर भिन्न max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f32_au(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न min, अचर भिन्न max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न min, अचर भिन्न max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न min, अचर भिन्न max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f32(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f32(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न min, अचर भिन्न max, अचर पूर्णाङ्क अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128) && hex_is_aligned((शून्य *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, अङ्क_elems);
    } अन्यथा {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, अङ्क_elems);
    }
}

#define HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_VectorPred pred_cap_दायाँ = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_VectorPred pred_cap_बायाँ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_दायाँ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_बायाँ, min_vec, tmp);                      \
    })

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f16_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 min, अचर _Float16 max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, परिमाण(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f16_au(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 min, अचर _Float16 max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, परिमाण(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f16_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 min, अचर _Float16 max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, परिमाण(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f16_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 min, अचर _Float16 max, uint32_t n) {
    अचर HVX_Vector min_vec = hvx_vec_splat_f16(min);
    अचर HVX_Vector max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, परिमाण(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

स्थिर अन्तःस्थ शून्य hvx_clamp_scalar_f16(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 min, अचर _Float16 max, अचर पूर्णाङ्क अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128) && hex_is_aligned((शून्य *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, अङ्क_elems);
    } अन्यथा {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, अङ्क_elems);
    }
}

#define HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_VectorPred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_Vector     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

स्थिर अन्तःस्थ शून्य hvx_leaky_relu_scalar_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न ns, uint32_t n) {
    अचर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    अचर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_leaky_relu_scalar_f32_au(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न ns, uint32_t n) {
    अचर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    अचर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_leaky_relu_scalar_f32_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न ns, uint32_t n) {
    अचर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    अचर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_leaky_relu_scalar_f32_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न ns, uint32_t n) {
    अचर HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    अचर HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, परिमाण(भिन्न), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

स्थिर अन्तःस्थ शून्य hvx_leaky_relu_scalar_f32(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न ns, अचर पूर्णाङ्क अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128) && hex_is_aligned((शून्य *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, अङ्क_elems);
    } अन्यथा यदि (hex_is_aligned((शून्य *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, अङ्क_elems);
    } अन्यथा {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, अङ्क_elems);
    }
}

//
// Abs
//

स्थिर अन्तःस्थ शून्य hvx_abs_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    अचर uint32_t elem_size = परिमाण(भिन्न);
    अचर uint32_t epv       = 128 / elem_size;
    अचर uint32_t nvec      = n / epv;
    अचर uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    लागि (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    यदि (nloe) {
        HVX_Vector v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((शून्य *) &vdst[i], nloe * elem_size, v);
    }
}

#define hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    गर {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        अचर uint32_t elem_size = परिमाण(_Float16);                       \
        अचर uint32_t epv  = 128 / elem_size;                             \
        अचर uint32_t nvec = n / epv;                                     \
        अचर uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        लागि (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        यदि (nloe) {                                                        \
            HVX_Vector v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } जबसम्म(0)

स्थिर अन्तःस्थ शून्य hvx_abs_f16_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_abs_f16_au(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_abs_f16_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_abs_f16_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_abs_f16(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर uint32_t अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128)) {
        यदि (hex_is_aligned((शून्य *) src, 128)) {
            hvx_abs_f16_aa(dst, src, अङ्क_elems);
        } अन्यथा {
            hvx_abs_f16_au(dst, src, अङ्क_elems);
        }
    } अन्यथा {
        यदि (hex_is_aligned((शून्य *) src, 128)) {
            hvx_abs_f16_ua(dst, src, अङ्क_elems);
        } अन्यथा {
            hvx_abs_f16_uu(dst, src, अङ्क_elems);
        }
    }
}

//
// Step
//

स्थिर अन्तःस्थ शून्य hvx_step_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    अचर uint32_t elem_size = परिमाण(भिन्न);
    अचर uint32_t epv       = 128 / elem_size;
    अचर uint32_t nvec      = n / epv;
    अचर uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    लागि (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f32(vsrc[i]);
    }
    यदि (nloe) {
        HVX_Vector v = hvx_vec_step_f32(vsrc[i]);
        hvx_vec_store_a((शून्य *) &vdst[i], nloe * elem_size, v);
    }
}

स्थिर अन्तःस्थ शून्य hvx_step_f16_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    अचर uint32_t elem_size = परिमाण(_Float16);
    अचर uint32_t epv       = 128 / elem_size;
    अचर uint32_t nvec      = n / epv;
    अचर uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    लागि (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f16(vsrc[i]);
    }
    यदि (nloe) {
        HVX_Vector v = hvx_vec_step_f16(vsrc[i]);
        hvx_vec_store_a((शून्य *) &vdst[i], nloe * elem_size, v);
    }
}

//
// Square
//

#define hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    गर {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        अचर uint32_t elem_size = परिमाण(भिन्न);                          \
        अचर uint32_t epv  = 128 / elem_size;                             \
        अचर uint32_t nvec = n / epv;                                     \
        अचर uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        लागि (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        यदि (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } जबसम्म(0)

स्थिर अन्तःस्थ शून्य hvx_sqr_f32_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f32_au(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f32_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f32_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f32(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर uint32_t अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128)) {
        यदि (hex_is_aligned((शून्य *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, अङ्क_elems);
        } अन्यथा {
            hvx_sqr_f32_au(dst, src, अङ्क_elems);
        }
    } अन्यथा {
        यदि (hex_is_aligned((शून्य *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, अङ्क_elems);
        } अन्यथा {
            hvx_sqr_f32_uu(dst, src, अङ्क_elems);
        }
    }
}

#define hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    गर {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        अचर uint32_t elem_size = परिमाण(_Float16);                       \
        अचर uint32_t epv  = 128 / elem_size;                             \
        अचर uint32_t nvec = n / epv;                                     \
        अचर uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        लागि (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        यदि (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } जबसम्म(0)

स्थिर अन्तःस्थ शून्य hvx_sqr_f16_aa(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f16_au(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f16_ua(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f16_uu(uint8_t * restrict dst, अचर uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_sqr_f16(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर uint32_t अङ्क_elems) {
    यदि (hex_is_aligned((शून्य *) dst, 128)) {
        यदि (hex_is_aligned((शून्य *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, अङ्क_elems);
        } अन्यथा {
            hvx_sqr_f16_au(dst, src, अङ्क_elems);
        }
    } अन्यथा {
        यदि (hex_is_aligned((शून्य *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, अङ्क_elems);
        } अन्यथा {
            hvx_sqr_f16_uu(dst, src, अङ्क_elems);
        }
    }
}

#undef HVX_OP_ADD_F32
#undef HVX_OP_SUB_F32
#undef HVX_OP_MUL_F32
#undef HVX_OP_ADD_F16
#undef HVX_OP_SUB_F16
#undef HVX_OP_MUL_F16
#undef hvx_arith_loop_body
#undef HVX_OP_ADD_SCALAR_F32
#undef HVX_OP_SUB_SCALAR_F32
#undef HVX_OP_MUL_SCALAR_F32
#undef HVX_OP_ADD_SCALAR_F16
#undef HVX_OP_SUB_SCALAR_F16
#undef HVX_OP_MUL_SCALAR_F16
#undef hvx_scalar_loop_body
#undef HVX_OP_MIN_SCALAR
#undef HVX_OP_MAX_SCALAR
#undef HVX_OP_MIN_SCALAR_F16
#undef HVX_OP_CLAMP_SCALAR
#undef HVX_OP_CLAMP_SCALAR_F16
#undef HVX_OP_LEAKY_RELU_SCALAR
#undef DEFINE_HVX_द्विआधारी_OP_VARIANTS
#undef DEFINE_HVX_द्विआधारी_SCALAR_OP_VARIANTS
#undef HVX_द्विआधारी_DISPATCHER
#undef UNUSED

#endif // HVX_ARITH_H
