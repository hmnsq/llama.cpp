#ifndef HVX_ARITH_H
#define HVX_ARITH_H

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <math.h>

#ଅନ୍ତର୍ଭାଵ "hvx-base.h"
#ଅନ୍ତର୍ଭାଵ "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#define UNUSED(x) (ଶୂନ୍ୟ)(x)

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_size, vec_store, vec_op) \
    କର {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        ଅଚଳ uint32_t epv  = 128 / (elem_size);                                          \
        ଅଚଳ uint32_t nvec = n / epv;                                                    \
        ଅଚଳ uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ପାଇଁ (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        ଯଦି (nloe) {                                                                       \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * (elem_size), v);                          \
        }                                                                                 \
    } ଯେତେବେଳେ(0)

#ଯଦି __HVX_ARCH__ < 79

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#ଅନ୍ୟଥା

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#endif

#define HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#define HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#define HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_aaa(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                       \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                      \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, ଆକାର(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_aau(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                       \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, ଆକାର(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_aua(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                       \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, ଆକାର(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_auu(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, ଆକାର(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uaa(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                      \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, ଆକାର(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uau(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, ଆକାର(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uua(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, ଆକାର(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uuu(uint8_t * dst, ଅଚଳ uint8_t * src0, ଅଚଳ uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, ଆକାର(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, ଦଶମିକ)
DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, ଦଶମିକ)
DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, ଦଶମିକ)

DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#define HVX_ଦ୍ୱିମିକ_DISPATCHER(OP_NAME)                                                                                                       \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, ଅଚଳ uint32_t ଅଙ୍କ_elems) { \
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {                                                                                                 \
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src0, 128)) {                                                                                            \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_aau(dst, src0, src1, ଅଙ୍କ_elems);                                               \
        } ଅନ୍ୟଥା {                                                                                                                             \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_auu(dst, src0, src1, ଅଙ୍କ_elems);                                               \
        }                                                                                                                                    \
    } ଅନ୍ୟଥା {                                                                                                                                 \
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src0, 128)) {                                                                                            \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_uau(dst, src0, src1, ଅଙ୍କ_elems);                                               \
        } ଅନ୍ୟଥା {                                                                                                                             \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_uuu(dst, src0, src1, ଅଙ୍କ_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_ଦ୍ୱିମିକ_DISPATCHER(hvx_add_f32)
HVX_ଦ୍ୱିମିକ_DISPATCHER(hvx_sub_f32)
HVX_ଦ୍ୱିମିକ_DISPATCHER(hvx_mul_f32)

HVX_ଦ୍ୱିମିକ_DISPATCHER(hvx_add_f16)
HVX_ଦ୍ୱିମିକ_DISPATCHER(hvx_sub_f16)
HVX_ଦ୍ୱିମିକ_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_mul_mul_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, ଅଚଳ uint8_t * restrict src2, ଅଚଳ uint32_t ଅଙ୍କ_elems) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src0 % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src1 % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    ଅଚଳ uint32_t elem_size = ଆକାର(ଦଶମିକ);
    ଅଚଳ uint32_t epv  = 128 / elem_size;
    ଅଚଳ uint32_t nvec = ଅଙ୍କ_elems / epv;
    ଅଚଳ uint32_t nloe = ଅଙ୍କ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ପାଇଁ (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    ଯଦି (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v2);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_mul_mul_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, ଅଚଳ uint8_t * restrict src2, ଅଚଳ uint32_t ଅଙ୍କ_elems) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src0 % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src1 % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    ଅଚଳ uint32_t elem_size = ଆକାର(_Float16);
    ଅଚଳ uint32_t epv  = 128 / elem_size;
    ଅଚଳ uint32_t nvec = ଅଙ୍କ_elems / epv;
    ଅଚଳ uint32_t nloe = ଅଙ୍କ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ପାଇଁ (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    ଯଦି (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, elem_size, vec_store, scalar_op_macro) \
    କର {                                                                                \
        dst_type * restrict vdst = (dst_type *) dst;                                    \
        src_type * restrict vsrc = (src_type *) src;                                    \
                                                                                        \
        ଅଚଳ uint32_t epv  = 128 / (elem_size);                                        \
        ଅଚଳ uint32_t nvec = n / epv;                                                  \
        ଅଚଳ uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        ପାଇଁ (; i < nvec; i++) {                                                         \
            HVX_Vector v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        ଯଦି (nloe) {                                                                     \
            HVX_Vector v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * (elem_size), v);                        \
        }                                                                               \
    } ଯେତେବେଳେ(0)

#define HVX_OP_ADD_SCALAR_F32(v)                                   \
    ({                                                             \
        ଅଚଳ HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#define HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#define HVX_OP_ADD_SCALAR_F16(v)                                   \
    ({                                                             \
        ଅଚଳ HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#define HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ELEM_TYPE val, uint32_t n) { \
    ଅଚଳ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ଅଚଳ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                                  \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ଆକାର(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ELEM_TYPE val, uint32_t n) { \
    ଅଚଳ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ଅଚଳ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ଆକାର(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ELEM_TYPE val, uint32_t n) { \
    ଅଚଳ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ଅଚଳ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ଆକାର(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ELEM_TYPE val, uint32_t n) { \
    ଅଚଳ HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    ଅଚଳ HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ଆକାର(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, ଦଶମିକ)
DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, ଦଶମିକ)
DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, ଦଶମିକ)

DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#define HVX_ଦ୍ୱିମିକ_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ELEM_TYPE val, ଅଚଳ uint32_t ଅଙ୍କ_elems) { \
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128) && hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, ଅଙ୍କ_elems);                                                                                   \
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, ଅଙ୍କ_elems);                                                                                   \
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, ଅଙ୍କ_elems);                                                                                   \
    } ଅନ୍ୟଥା {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, ଅଙ୍କ_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_ଦ୍ୱିମିକ_SCALAR_DISPATCHER(hvx_add_scalar_f32, ଦଶମିକ)
HVX_ଦ୍ୱିମିକ_SCALAR_DISPATCHER(hvx_sub_scalar_f32, ଦଶମିକ)
HVX_ଦ୍ୱିମିକ_SCALAR_DISPATCHER(hvx_mul_scalar_f32, ଦଶମିକ)

HVX_ଦ୍ୱିମିକ_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_ଦ୍ୱିମିକ_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_ଦ୍ୱିମିକ_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f32_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f32_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f32_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f32(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128) && hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା {
        hvx_min_scalar_f32_uu(dst, src, val, ଅଙ୍କ_elems);
    }
}

// MAX Scalar variants

#define HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_max_scalar_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_max_scalar_f32_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_max_scalar_f32_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_max_scalar_f32_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_max_scalar_f32(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ val, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128) && hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା {
        hvx_max_scalar_f32_uu(dst, src, val, ଅଙ୍କ_elems);
    }
}

#define HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ଆକାର(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f16_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ଆକାର(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f16_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ଆକାର(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f16_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ଆକାର(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_min_scalar_f16(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128) && hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା {
        hvx_min_scalar_f16_uu(dst, src, val, ଅଙ୍କ_elems);
    }
}

// CLAMP Scalar variants

#define HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_VectorPred pred_cap_ଡାହାଣ = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_ବାମ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_ଡାହାଣ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ବାମ, min_vec, tmp);                      \
    })

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ min, ଅଚଳ ଦଶମିକ max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f32_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ min, ଅଚଳ ଦଶମିକ max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ min, ଅଚଳ ଦଶମିକ max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ min, ଅଚଳ ଦଶମିକ max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f32(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f32(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ min, ଅଚଳ ଦଶମିକ max, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128) && hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, ଅଙ୍କ_elems);
    }
}

#define HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_VectorPred pred_cap_ଡାହାଣ = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_VectorPred pred_cap_ବାମ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_ଡାହାଣ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ବାମ, min_vec, tmp);                      \
    })

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 min, ଅଚଳ _Float16 max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ଆକାର(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f16_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 min, ଅଚଳ _Float16 max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ଆକାର(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f16_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 min, ଅଚଳ _Float16 max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ଆକାର(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f16_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 min, ଅଚଳ _Float16 max, uint32_t n) {
    ଅଚଳ HVX_Vector min_vec = hvx_vec_splat_f16(min);
    ଅଚଳ HVX_Vector max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ଆକାର(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_clamp_scalar_f16(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 min, ଅଚଳ _Float16 max, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128) && hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, ଅଙ୍କ_elems);
    }
}

#define HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_VectorPred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_Vector     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_leaky_relu_scalar_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ ns, uint32_t n) {
    ଅଚଳ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ଅଚଳ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_leaky_relu_scalar_f32_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ ns, uint32_t n) {
    ଅଚଳ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ଅଚଳ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_leaky_relu_scalar_f32_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ ns, uint32_t n) {
    ଅଚଳ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ଅଚଳ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_leaky_relu_scalar_f32_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ ns, uint32_t n) {
    ଅଚଳ HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    ଅଚଳ HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ଆକାର(ଦଶମିକ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_leaky_relu_scalar_f32(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ଦଶମିକ ns, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128) && hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, ଅଙ୍କ_elems);
    } ଅନ୍ୟଥା {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, ଅଙ୍କ_elems);
    }
}

//
// Abs
//

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_abs_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ଅଚଳ uint32_t elem_size = ଆକାର(ଦଶମିକ);
    ଅଚଳ uint32_t epv       = 128 / elem_size;
    ଅଚଳ uint32_t nvec      = n / epv;
    ଅଚଳ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ପାଇଁ (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    ଯଦି (nloe) {
        HVX_Vector v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v);
    }
}

#define hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    କର {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ଅଚଳ uint32_t elem_size = ଆକାର(_Float16);                       \
        ଅଚଳ uint32_t epv  = 128 / elem_size;                             \
        ଅଚଳ uint32_t nvec = n / epv;                                     \
        ଅଚଳ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ପାଇଁ (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        ଯଦି (nloe) {                                                        \
            HVX_Vector v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ଯେତେବେଳେ(0)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_abs_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_abs_f16_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_abs_f16_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_abs_f16_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_abs_f16(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ uint32_t ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
            hvx_abs_f16_aa(dst, src, ଅଙ୍କ_elems);
        } ଅନ୍ୟଥା {
            hvx_abs_f16_au(dst, src, ଅଙ୍କ_elems);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
            hvx_abs_f16_ua(dst, src, ଅଙ୍କ_elems);
        } ଅନ୍ୟଥା {
            hvx_abs_f16_uu(dst, src, ଅଙ୍କ_elems);
        }
    }
}

//
// Step
//

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_step_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ଅଚଳ uint32_t elem_size = ଆକାର(ଦଶମିକ);
    ଅଚଳ uint32_t epv       = 128 / elem_size;
    ଅଚଳ uint32_t nvec      = n / epv;
    ଅଚଳ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ପାଇଁ (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f32(vsrc[i]);
    }
    ଯଦି (nloe) {
        HVX_Vector v = hvx_vec_step_f32(vsrc[i]);
        hvx_vec_store_a((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_step_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ଅଚଳ uint32_t elem_size = ଆକାର(_Float16);
    ଅଚଳ uint32_t epv       = 128 / elem_size;
    ଅଚଳ uint32_t nvec      = n / epv;
    ଅଚଳ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ପାଇଁ (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f16(vsrc[i]);
    }
    ଯଦି (nloe) {
        HVX_Vector v = hvx_vec_step_f16(vsrc[i]);
        hvx_vec_store_a((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v);
    }
}

//
// Square
//

#define hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    କର {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ଅଚଳ uint32_t elem_size = ଆକାର(ଦଶମିକ);                          \
        ଅଚଳ uint32_t epv  = 128 / elem_size;                             \
        ଅଚଳ uint32_t nvec = n / epv;                                     \
        ଅଚଳ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ପାଇଁ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ଯଦି (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ଯେତେବେଳେ(0)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f32_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f32_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f32_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f32(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ uint32_t ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, ଅଙ୍କ_elems);
        } ଅନ୍ୟଥା {
            hvx_sqr_f32_au(dst, src, ଅଙ୍କ_elems);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, ଅଙ୍କ_elems);
        } ଅନ୍ୟଥା {
            hvx_sqr_f32_uu(dst, src, ଅଙ୍କ_elems);
        }
    }
}

#define hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    କର {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        ଅଚଳ uint32_t elem_size = ଆକାର(_Float16);                       \
        ଅଚଳ uint32_t epv  = 128 / elem_size;                             \
        ଅଚଳ uint32_t nvec = n / epv;                                     \
        ଅଚଳ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ପାଇଁ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ଯଦି (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ଯେତେବେଳେ(0)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f16_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f16_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    ପ୍ରତିଜ୍ଞା((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f16_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_sqr_f16(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ uint32_t ଅଙ୍କ_elems) {
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, ଅଙ୍କ_elems);
        } ଅନ୍ୟଥା {
            hvx_sqr_f16_au(dst, src, ଅଙ୍କ_elems);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, ଅଙ୍କ_elems);
        } ଅନ୍ୟଥା {
            hvx_sqr_f16_uu(dst, src, ଅଙ୍କ_elems);
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
#undef DEFINE_HVX_ଦ୍ୱିମିକ_OP_VARIANTS
#undef DEFINE_HVX_ଦ୍ୱିମିକ_SCALAR_OP_VARIANTS
#undef HVX_ଦ୍ୱିମିକ_DISPATCHER
#undef UNUSED

#endif // HVX_ARITH_H
