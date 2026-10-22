#ifndef HVX_ARITH_H
#define HVX_ARITH_H

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <math.h>

#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#define UNUSED(x) (ಶೂನ್ಯ)(x)

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_size, vec_store, vec_op) \
    ಕುರು {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        ಧ್ರುವ uint32_t epv  = 128 / (elem_size);                                          \
        ಧ್ರುವ uint32_t nvec = n / epv;                                                    \
        ಧ್ರುವ uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ಪ್ರತಿ (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        ಯದಿ (nloe) {                                                                       \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * (elem_size), v);                          \
        }                                                                                 \
    } ಯಾವತ್(0)

#ಯದಿ __HVX_ARCH__ < 79

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#ಅನ್ಯಥಾ

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#endif

#define HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#define HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#define HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_aaa(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_aau(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_aua(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_auu(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_uaa(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_uau(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_uua(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_uuu(uint8_t * dst, ಧ್ರುವ uint8_t * src0, ಧ್ರುವ uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, ಭಿನ್ನ)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, ಭಿನ್ನ)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, ಭಿನ್ನ)

DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#define HVX_ದ್ವಿಮಾನ_DISPATCHER(OP_NAME)                                                                                                       \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src0, ಧ್ರುವ uint8_t * restrict src1, ಧ್ರುವ uint32_t ಅಂಕೆ_elems) { \
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {                                                                                                 \
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src0, 128)) {                                                                                            \
            ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಅನ್ಯಥಾ                                    OP_NAME##_aau(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        } ಅನ್ಯಥಾ {                                                                                                                             \
            ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಅನ್ಯಥಾ                                    OP_NAME##_auu(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        }                                                                                                                                    \
    } ಅನ್ಯಥಾ {                                                                                                                                 \
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src0, 128)) {                                                                                            \
            ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಅನ್ಯಥಾ                                    OP_NAME##_uau(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        } ಅನ್ಯಥಾ {                                                                                                                             \
            ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಅನ್ಯಥಾ                                    OP_NAME##_uuu(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_add_f32)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_sub_f32)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_mul_f32)

HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_add_f16)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_sub_f16)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_mul_mul_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src0, ಧ್ರುವ uint8_t * restrict src1, ಧ್ರುವ uint8_t * restrict src2, ಧ್ರುವ uint32_t ಅಂಕೆ_elems) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src1 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t epv  = 128 / elem_size;
    ಧ್ರುವ uint32_t nvec = ಅಂಕೆ_elems / epv;
    ಧ್ರುವ uint32_t nloe = ಅಂಕೆ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಪ್ರತಿ (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    ಯದಿ (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v2);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_mul_mul_f16_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src0, ಧ್ರುವ uint8_t * restrict src1, ಧ್ರುವ uint8_t * restrict src2, ಧ್ರುವ uint32_t ಅಂಕೆ_elems) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src1 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(_Float16);
    ಧ್ರುವ uint32_t epv  = 128 / elem_size;
    ಧ್ರುವ uint32_t nvec = ಅಂಕೆ_elems / epv;
    ಧ್ರುವ uint32_t nloe = ಅಂಕೆ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಪ್ರತಿ (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    ಯದಿ (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, elem_size, vec_store, scalar_op_macro) \
    ಕುರು {                                                                                \
        dst_type * restrict vdst = (dst_type *) dst;                                    \
        src_type * restrict vsrc = (src_type *) src;                                    \
                                                                                        \
        ಧ್ರುವ uint32_t epv  = 128 / (elem_size);                                        \
        ಧ್ರುವ uint32_t nvec = n / epv;                                                  \
        ಧ್ರುವ uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        ಪ್ರತಿ (; i < nvec; i++) {                                                         \
            HVX_Vector v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        ಯದಿ (nloe) {                                                                     \
            HVX_Vector v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * (elem_size), v);                        \
        }                                                                               \
    } ಯಾವತ್(0)

#define HVX_OP_ADD_SCALAR_F32(v)                                   \
    ({                                                             \
        ಧ್ರುವ HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#define HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#define HVX_OP_ADD_SCALAR_F16(v)                                   \
    ({                                                             \
        ಧ್ರುವ HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#define HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ELEM_TYPE val, uint32_t n) { \
    ಧ್ರುವ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ಧ್ರುವ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                  \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ELEM_TYPE val, uint32_t n) { \
    ಧ್ರುವ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ಧ್ರುವ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ELEM_TYPE val, uint32_t n) { \
    ಧ್ರುವ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ಧ್ರುವ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME##_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ELEM_TYPE val, uint32_t n) { \
    ಧ್ರುವ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ಧ್ರುವ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, ಭಿನ್ನ)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, ಭಿನ್ನ)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, ಭಿನ್ನ)

DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#define HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ OP_NAME(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ELEM_TYPE val, ಧ್ರುವ uint32_t ಅಂಕೆ_elems) { \
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    } ಅನ್ಯಥಾ {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_add_scalar_f32, ಭಿನ್ನ)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_sub_scalar_f32, ಭಿನ್ನ)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_mul_scalar_f32, ಭಿನ್ನ)

HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f32_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f32_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f32_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f32(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ {
        hvx_min_scalar_f32_uu(dst, src, val, ಅಂಕೆ_elems);
    }
}

// MAX Scalar variants

#define HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_max_scalar_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_max_scalar_f32_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_max_scalar_f32_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_max_scalar_f32_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_max_scalar_f32(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ val, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ {
        hvx_max_scalar_f32_uu(dst, src, val, ಅಂಕೆ_elems);
    }
}

#define HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f16_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f16_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f16_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f16_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 val, uint32_t n) {
    ಧ್ರುವ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_min_scalar_f16(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 val, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ {
        hvx_min_scalar_f16_uu(dst, src, val, ಅಂಕೆ_elems);
    }
}

// CLAMP Scalar variants

#define HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_VectorPred pred_cap_ಬಲ = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_ಎಡ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_ಬಲ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ಎಡ, min_vec, tmp);                      \
    })

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ min, ಧ್ರುವ ಭಿನ್ನ max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f32_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ min, ಧ್ರುವ ಭಿನ್ನ max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ min, ಧ್ರುವ ಭಿನ್ನ max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ min, ಧ್ರುವ ಭಿನ್ನ max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f32(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ min, ಧ್ರುವ ಭಿನ್ನ max, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, ಅಂಕೆ_elems);
    }
}

#define HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_VectorPred pred_cap_ಬಲ = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_VectorPred pred_cap_ಎಡ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_ಬಲ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ಎಡ, min_vec, tmp);                      \
    })

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f16_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 min, ಧ್ರುವ _Float16 max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f16_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 min, ಧ್ರುವ _Float16 max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f16_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 min, ಧ್ರುವ _Float16 max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f16_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 min, ಧ್ರುವ _Float16 max, uint32_t n) {
    ಧ್ರುವ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ಧ್ರುವ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_clamp_scalar_f16(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ _Float16 min, ಧ್ರುವ _Float16 max, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, ಅಂಕೆ_elems);
    }
}

#define HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_VectorPred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_Vector     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ ns, uint32_t n) {
    ಧ್ರುವ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ಧ್ರುವ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ ns, uint32_t n) {
    ಧ್ರುವ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ಧ್ರುವ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ ns, uint32_t n) {
    ಧ್ರುವ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ಧ್ರುವ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ ns, uint32_t n) {
    ಧ್ರುವ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ಧ್ರುವ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ ಭಿನ್ನ ns, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, ಅಂಕೆ_elems);
    } ಅನ್ಯಥಾ {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, ಅಂಕೆ_elems);
    }
}

//
// Abs
//

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_abs_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t epv       = 128 / elem_size;
    ಧ್ರುವ uint32_t nvec      = n / epv;
    ಧ್ರುವ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಪ್ರತಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    ಯದಿ (nloe) {
        HVX_Vector v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v);
    }
}

#define hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    ಕುರು {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(_Float16);                       \
        ಧ್ರುವ uint32_t epv  = 128 / elem_size;                             \
        ಧ್ರುವ uint32_t nvec = n / epv;                                     \
        ಧ್ರುವ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ಪ್ರತಿ (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        ಯದಿ (nloe) {                                                        \
            HVX_Vector v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ಯಾವತ್(0)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_abs_f16_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_abs_f16_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_abs_f16_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_abs_f16_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_abs_f16(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ uint32_t ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_abs_f16_aa(dst, src, ಅಂಕೆ_elems);
        } ಅನ್ಯಥಾ {
            hvx_abs_f16_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_abs_f16_ua(dst, src, ಅಂಕೆ_elems);
        } ಅನ್ಯಥಾ {
            hvx_abs_f16_uu(dst, src, ಅಂಕೆ_elems);
        }
    }
}

//
// Step
//

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_step_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t epv       = 128 / elem_size;
    ಧ್ರುವ uint32_t nvec      = n / epv;
    ಧ್ರುವ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಪ್ರತಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f32(vsrc[i]);
    }
    ಯದಿ (nloe) {
        HVX_Vector v = hvx_vec_step_f32(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_step_f16_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(_Float16);
    ಧ್ರುವ uint32_t epv       = 128 / elem_size;
    ಧ್ರುವ uint32_t nvec      = n / epv;
    ಧ್ರುವ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಪ್ರತಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f16(vsrc[i]);
    }
    ಯದಿ (nloe) {
        HVX_Vector v = hvx_vec_step_f16(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v);
    }
}

//
// Square
//

#define hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    ಕುರು {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(ಭಿನ್ನ);                          \
        ಧ್ರುವ uint32_t epv  = 128 / elem_size;                             \
        ಧ್ರುವ uint32_t nvec = n / epv;                                     \
        ಧ್ರುವ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ಪ್ರತಿ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ಯದಿ (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ಯಾವತ್(0)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f32_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f32_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f32_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f32_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f32(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ uint32_t ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, ಅಂಕೆ_elems);
        } ಅನ್ಯಥಾ {
            hvx_sqr_f32_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, ಅಂಕೆ_elems);
        } ಅನ್ಯಥಾ {
            hvx_sqr_f32_uu(dst, src, ಅಂಕೆ_elems);
        }
    }
}

#define hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    ಕುರು {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ಧ್ರುವ uint32_t elem_size = ಪರಿಮಾಣ(_Float16);                       \
        ಧ್ರುವ uint32_t epv  = 128 / elem_size;                             \
        ಧ್ರುವ uint32_t nvec = n / epv;                                     \
        ಧ್ರುವ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ಪ್ರತಿ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ಯದಿ (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ಯಾವತ್(0)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f16_aa(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f16_au(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f16_ua(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f16_uu(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_sqr_f16(uint8_t * restrict dst, ಧ್ರುವ uint8_t * restrict src, ಧ್ರುವ uint32_t ಅಂಕೆ_elems) {
    ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, ಅಂಕೆ_elems);
        } ಅನ್ಯಥಾ {
            hvx_sqr_f16_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, ಅಂಕೆ_elems);
        } ಅನ್ಯಥಾ {
            hvx_sqr_f16_uu(dst, src, ಅಂಕೆ_elems);
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
#undef DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS
#undef DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS
#undef HVX_ದ್ವಿಮಾನ_DISPATCHER
#undef UNUSED

#endif // HVX_ARITH_H
