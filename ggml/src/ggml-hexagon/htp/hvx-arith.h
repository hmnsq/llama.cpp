#ifndef HVX_ARITH_H
#define HVX_ARITH_H

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <math.h>

#অন্তর্ভাব "hvx-base.h"
#অন্তর্ভাব "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#define UNUSED(x) (শূন্য)(x)

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_আকার, vec_store, vec_op) \
    করো {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        ধ্রুবক uint32_t epv  = 128 / (elem_আকার);                                          \
        ধ্রুবক uint32_t nvec = n / epv;                                                    \
        ধ্রুবক uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        জন্য (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        যদি (nloe) {                                                                       \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((শূন্য *) &vdst[i], nloe * (elem_আকার), v);                          \
        }                                                                                 \
    } যখন(0)

#যদি __HVX_ARCH__ < 79

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#অন্যথা

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#endif

#define HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#define HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#define HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_দ্বিমিক_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
স্থির অন্তঃরেখ শূন্য OP_NAME##_aaa(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    প্রতিজ্ঞা((uintptr_t) dst % 128 == 0);                                                                       \
    প্রতিজ্ঞা((uintptr_t) src0 % 128 == 0);                                                                      \
    প্রতিজ্ঞা((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, আকার(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
স্থির অন্তঃরেখ শূন্য OP_NAME##_aau(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    প্রতিজ্ঞা((uintptr_t) dst % 128 == 0);                                                                       \
    প্রতিজ্ঞা((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, আকার(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
স্থির অন্তঃরেখ শূন্য OP_NAME##_aua(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    প্রতিজ্ঞা((uintptr_t) dst % 128 == 0);                                                                       \
    প্রতিজ্ঞা((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, আকার(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
স্থির অন্তঃরেখ শূন্য OP_NAME##_auu(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    প্রতিজ্ঞা((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, আকার(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
স্থির অন্তঃরেখ শূন্য OP_NAME##_uaa(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    প্রতিজ্ঞা((uintptr_t) src0 % 128 == 0);                                                                      \
    প্রতিজ্ঞা((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, আকার(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
স্থির অন্তঃরেখ শূন্য OP_NAME##_uau(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    প্রতিজ্ঞা((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, আকার(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
স্থির অন্তঃরেখ শূন্য OP_NAME##_uua(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    প্রতিজ্ঞা((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, আকার(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
স্থির অন্তঃরেখ শূন্য OP_NAME##_uuu(uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, আকার(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_দ্বিমিক_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, দশমিক)
DEFINE_HVX_দ্বিমিক_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, দশমিক)
DEFINE_HVX_দ্বিমিক_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, দশমিক)

DEFINE_HVX_দ্বিমিক_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_দ্বিমিক_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_দ্বিমিক_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#define HVX_দ্বিমিক_DISPATCHER(OP_NAME)                                                                                                       \
স্থির অন্তঃরেখ শূন্য OP_NAME(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src0, ধ্রুবক uint8_t * restrict src1, ধ্রুবক uint32_t অঙ্ক_elems) { \
    যদি (hex_is_aligned((শূন্য *) dst, 128)) {                                                                                                 \
        যদি (hex_is_aligned((শূন্য *) src0, 128)) {                                                                                            \
            যদি (hex_is_aligned((শূন্য *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, অঙ্ক_elems);                                               \
            অন্যথা                                    OP_NAME##_aau(dst, src0, src1, অঙ্ক_elems);                                               \
        } অন্যথা {                                                                                                                             \
            যদি (hex_is_aligned((শূন্য *) src1, 128)) OP_NAME##_aua(dst, src0, src1, অঙ্ক_elems);                                               \
            অন্যথা                                    OP_NAME##_auu(dst, src0, src1, অঙ্ক_elems);                                               \
        }                                                                                                                                    \
    } অন্যথা {                                                                                                                                 \
        যদি (hex_is_aligned((শূন্য *) src0, 128)) {                                                                                            \
            যদি (hex_is_aligned((শূন্য *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, অঙ্ক_elems);                                               \
            অন্যথা                                    OP_NAME##_uau(dst, src0, src1, অঙ্ক_elems);                                               \
        } অন্যথা {                                                                                                                             \
            যদি (hex_is_aligned((শূন্য *) src1, 128)) OP_NAME##_uua(dst, src0, src1, অঙ্ক_elems);                                               \
            অন্যথা                                    OP_NAME##_uuu(dst, src0, src1, অঙ্ক_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_দ্বিমিক_DISPATCHER(hvx_add_f32)
HVX_দ্বিমিক_DISPATCHER(hvx_sub_f32)
HVX_দ্বিমিক_DISPATCHER(hvx_mul_f32)

HVX_দ্বিমিক_DISPATCHER(hvx_add_f16)
HVX_দ্বিমিক_DISPATCHER(hvx_sub_f16)
HVX_দ্বিমিক_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
স্থির অন্তঃরেখ শূন্য hvx_mul_mul_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src0, ধ্রুবক uint8_t * restrict src1, ধ্রুবক uint8_t * restrict src2, ধ্রুবক uint32_t অঙ্ক_elems) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src0 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src1 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    ধ্রুবক uint32_t elem_আকার = আকার(দশমিক);
    ধ্রুবক uint32_t epv  = 128 / elem_আকার;
    ধ্রুবক uint32_t nvec = অঙ্ক_elems / epv;
    ধ্রুবক uint32_t nloe = অঙ্ক_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    যদি (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * elem_আকার, v2);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_mul_mul_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src0, ধ্রুবক uint8_t * restrict src1, ধ্রুবক uint8_t * restrict src2, ধ্রুবক uint32_t অঙ্ক_elems) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src0 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src1 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    ধ্রুবক uint32_t elem_আকার = আকার(_Float16);
    ধ্রুবক uint32_t epv  = 128 / elem_আকার;
    ধ্রুবক uint32_t nvec = অঙ্ক_elems / epv;
    ধ্রুবক uint32_t nloe = অঙ্ক_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    যদি (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * elem_আকার, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, elem_আকার, vec_store, scalar_op_macro) \
    করো {                                                                                \
        dst_type * restrict vdst = (dst_type *) dst;                                    \
        src_type * restrict vsrc = (src_type *) src;                                    \
                                                                                        \
        ধ্রুবক uint32_t epv  = 128 / (elem_আকার);                                        \
        ধ্রুবক uint32_t nvec = n / epv;                                                  \
        ধ্রুবক uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        জন্য (; i < nvec; i++) {                                                         \
            HVX_Vector v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        যদি (nloe) {                                                                     \
            HVX_Vector v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((শূন্য *) &vdst[i], nloe * (elem_আকার), v);                        \
        }                                                                               \
    } যখন(0)

#define HVX_OP_ADD_SCALAR_F32(v)                                   \
    ({                                                             \
        ধ্রুবক HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#define HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#define HVX_OP_ADD_SCALAR_F16(v)                                   \
    ({                                                             \
        ধ্রুবক HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#define HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
স্থির অন্তঃরেখ শূন্য OP_NAME##_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক ELEM_TYPE val, uint32_t n) { \
    ধ্রুবক HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ধ্রুবক HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    প্রতিজ্ঞা((uintptr_t) dst % 128 == 0);                                                                                  \
    প্রতিজ্ঞা((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, আকার(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
স্থির অন্তঃরেখ শূন্য OP_NAME##_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক ELEM_TYPE val, uint32_t n) { \
    ধ্রুবক HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ধ্রুবক HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    প্রতিজ্ঞা((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, আকার(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
স্থির অন্তঃরেখ শূন্য OP_NAME##_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক ELEM_TYPE val, uint32_t n) { \
    ধ্রুবক HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ধ্রুবক HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    প্রতিজ্ঞা((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, আকার(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
স্থির অন্তঃরেখ শূন্য OP_NAME##_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক ELEM_TYPE val, uint32_t n) { \
    ধ্রুবক HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ধ্রুবক HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, আকার(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, দশমিক)
DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, দশমিক)
DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, দশমিক)

DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#define HVX_দ্বিমিক_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
স্থির অন্তঃরেখ শূন্য OP_NAME(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক ELEM_TYPE val, ধ্রুবক uint32_t অঙ্ক_elems) { \
    যদি (hex_is_aligned((শূন্য *) dst, 128) && hex_is_aligned((শূন্য *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, অঙ্ক_elems);                                                                                   \
    } অন্যথা যদি (hex_is_aligned((শূন্য *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, অঙ্ক_elems);                                                                                   \
    } অন্যথা যদি (hex_is_aligned((শূন্য *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, অঙ্ক_elems);                                                                                   \
    } অন্যথা {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, অঙ্ক_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_দ্বিমিক_SCALAR_DISPATCHER(hvx_add_scalar_f32, দশমিক)
HVX_দ্বিমিক_SCALAR_DISPATCHER(hvx_sub_scalar_f32, দশমিক)
HVX_দ্বিমিক_SCALAR_DISPATCHER(hvx_mul_scalar_f32, দশমিক)

HVX_দ্বিমিক_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_দ্বিমিক_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_দ্বিমিক_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128) && hex_is_aligned((শূন্য *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, অঙ্ক_elems);
    } অন্যথা {
        hvx_min_scalar_f32_uu(dst, src, val, অঙ্ক_elems);
    }
}

// MAX Scalar variants

#define HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

স্থির অন্তঃরেখ শূন্য hvx_max_scalar_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_max_scalar_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_max_scalar_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_max_scalar_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_max_scalar_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128) && hex_is_aligned((শূন্য *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, অঙ্ক_elems);
    } অন্যথা {
        hvx_max_scalar_f32_uu(dst, src, val, অঙ্ক_elems);
    }
}

#define HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f16(val);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, আকার(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f16_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f16(val);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, আকার(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f16_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f16(val);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, আকার(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f16_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 val, uint32_t n) {
    ধ্রুবক HVX_Vector val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, আকার(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_min_scalar_f16(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 val, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128) && hex_is_aligned((শূন্য *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, অঙ্ক_elems);
    } অন্যথা {
        hvx_min_scalar_f16_uu(dst, src, val, অঙ্ক_elems);
    }
}

// CLAMP Scalar variants

#define HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_VectorPred pred_cap_ডান = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_বাম  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_ডান, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_বাম, min_vec, tmp);                      \
    })

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক min, ধ্রুবক দশমিক max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f32(max);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক min, ধ্রুবক দশমিক max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f32(max);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক min, ধ্রুবক দশমিক max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f32(max);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক min, ধ্রুবক দশমিক max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক min, ধ্রুবক দশমিক max, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128) && hex_is_aligned((শূন্য *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, অঙ্ক_elems);
    } অন্যথা {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, অঙ্ক_elems);
    }
}

#define HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_VectorPred pred_cap_ডান = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_VectorPred pred_cap_বাম  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_ডান, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_বাম, min_vec, tmp);                      \
    })

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 min, ধ্রুবক _Float16 max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f16(max);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, আকার(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f16_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 min, ধ্রুবক _Float16 max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f16(max);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, আকার(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f16_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 min, ধ্রুবক _Float16 max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f16(max);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, আকার(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f16_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 min, ধ্রুবক _Float16 max, uint32_t n) {
    ধ্রুবক HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ধ্রুবক HVX_Vector max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, আকার(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

স্থির অন্তঃরেখ শূন্য hvx_clamp_scalar_f16(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 min, ধ্রুবক _Float16 max, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128) && hex_is_aligned((শূন্য *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, অঙ্ক_elems);
    } অন্যথা {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, অঙ্ক_elems);
    }
}

#define HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_VectorPred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_Vector     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

স্থির অন্তঃরেখ শূন্য hvx_leaky_relu_scalar_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক ns, uint32_t n) {
    ধ্রুবক HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ধ্রুবক HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_leaky_relu_scalar_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক ns, uint32_t n) {
    ধ্রুবক HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ধ্রুবক HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, আকার(দশমিক), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_leaky_relu_scalar_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক ns, uint32_t n) {
    ধ্রুবক HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ধ্রুবক HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_leaky_relu_scalar_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক ns, uint32_t n) {
    ধ্রুবক HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ধ্রুবক HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, আকার(দশমিক), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

স্থির অন্তঃরেখ শূন্য hvx_leaky_relu_scalar_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক ns, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128) && hex_is_aligned((শূন্য *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, অঙ্ক_elems);
    } অন্যথা যদি (hex_is_aligned((শূন্য *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, অঙ্ক_elems);
    } অন্যথা {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, অঙ্ক_elems);
    }
}

//
// Abs
//

স্থির অন্তঃরেখ শূন্য hvx_abs_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ধ্রুবক uint32_t elem_আকার = আকার(দশমিক);
    ধ্রুবক uint32_t epv       = 128 / elem_আকার;
    ধ্রুবক uint32_t nvec      = n / epv;
    ধ্রুবক uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    যদি (nloe) {
        HVX_Vector v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * elem_আকার, v);
    }
}

#define hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    করো {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ধ্রুবক uint32_t elem_আকার = আকার(_Float16);                       \
        ধ্রুবক uint32_t epv  = 128 / elem_আকার;                             \
        ধ্রুবক uint32_t nvec = n / epv;                                     \
        ধ্রুবক uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        জন্য (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        যদি (nloe) {                                                        \
            HVX_Vector v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((শূন্য *) &vdst[i], nloe * elem_আকার, v);             \
        }                                                                  \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_abs_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_abs_f16_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_abs_f16_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_abs_f16_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_abs_f16(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক uint32_t অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        যদি (hex_is_aligned((শূন্য *) src, 128)) {
            hvx_abs_f16_aa(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_abs_f16_au(dst, src, অঙ্ক_elems);
        }
    } অন্যথা {
        যদি (hex_is_aligned((শূন্য *) src, 128)) {
            hvx_abs_f16_ua(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_abs_f16_uu(dst, src, অঙ্ক_elems);
        }
    }
}

//
// Step
//

স্থির অন্তঃরেখ শূন্য hvx_step_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ধ্রুবক uint32_t elem_আকার = আকার(দশমিক);
    ধ্রুবক uint32_t epv       = 128 / elem_আকার;
    ধ্রুবক uint32_t nvec      = n / epv;
    ধ্রুবক uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f32(vsrc[i]);
    }
    যদি (nloe) {
        HVX_Vector v = hvx_vec_step_f32(vsrc[i]);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * elem_আকার, v);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_step_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ধ্রুবক uint32_t elem_আকার = আকার(_Float16);
    ধ্রুবক uint32_t epv       = 128 / elem_আকার;
    ধ্রুবক uint32_t nvec      = n / epv;
    ধ্রুবক uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f16(vsrc[i]);
    }
    যদি (nloe) {
        HVX_Vector v = hvx_vec_step_f16(vsrc[i]);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * elem_আকার, v);
    }
}

//
// Square
//

#define hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    করো {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ধ্রুবক uint32_t elem_আকার = আকার(দশমিক);                          \
        ধ্রুবক uint32_t epv  = 128 / elem_আকার;                             \
        ধ্রুবক uint32_t nvec = n / epv;                                     \
        ধ্রুবক uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        জন্য (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        যদি (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((শূন্য *) &vdst[i], nloe * elem_আকার, v);             \
        }                                                                  \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_sqr_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক uint32_t অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        যদি (hex_is_aligned((শূন্য *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqr_f32_au(dst, src, অঙ্ক_elems);
        }
    } অন্যথা {
        যদি (hex_is_aligned((শূন্য *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqr_f32_uu(dst, src, অঙ্ক_elems);
        }
    }
}

#define hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    করো {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ধ্রুবক uint32_t elem_আকার = আকার(_Float16);                       \
        ধ্রুবক uint32_t epv  = 128 / elem_আকার;                             \
        ধ্রুবক uint32_t nvec = n / epv;                                     \
        ধ্রুবক uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        জন্য (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        যদি (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((শূন্য *) &vdst[i], nloe * elem_আকার, v);             \
        }                                                                  \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_sqr_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f16_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f16_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f16_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqr_f16(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক uint32_t অঙ্ক_elems) {
    যদি (hex_is_aligned((শূন্য *) dst, 128)) {
        যদি (hex_is_aligned((শূন্য *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqr_f16_au(dst, src, অঙ্ক_elems);
        }
    } অন্যথা {
        যদি (hex_is_aligned((শূন্য *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqr_f16_uu(dst, src, অঙ্ক_elems);
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
#undef DEFINE_HVX_দ্বিমিক_OP_VARIANTS
#undef DEFINE_HVX_দ্বিমিক_SCALAR_OP_VARIANTS
#undef HVX_দ্বিমিক_DISPATCHER
#undef UNUSED

#endif // HVX_ARITH_H
