#ifndef HVX_ARITH_H
#define HVX_ARITH_H

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <math.h>

#അന്തര്ഭാവ "hvx-base.h"
#അന്തര്ഭാവ "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#define UNUSED(x) (ശൂന്യം)(x)

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_size, vec_store, vec_op) \
    ചെയ്യുക {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        സ്ഥിരം uint32_t epv  = 128 / (elem_size);                                          \
        സ്ഥിരം uint32_t nvec = n / epv;                                                    \
        സ്ഥിരം uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        വേണ്ടി (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        എങ്കിൽ (nloe) {                                                                       \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((ശൂന്യം *) &vdst[i], nloe * (elem_size), v);                          \
        }                                                                                 \
    } ആകുമ്പോൾ(0)

#എങ്കിൽ __HVX_ARCH__ < 79

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#അല്ലെങ്കിൽ

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#endif

#define HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#define HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#define HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_ദ്വിമാന_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_aaa(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    പ്രതിജ്ഞാ((uintptr_t) dst % 128 == 0);                                                                       \
    പ്രതിജ്ഞാ((uintptr_t) src0 % 128 == 0);                                                                      \
    പ്രതിജ്ഞാ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_aau(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    പ്രതിജ്ഞാ((uintptr_t) dst % 128 == 0);                                                                       \
    പ്രതിജ്ഞാ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_aua(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    പ്രതിജ്ഞാ((uintptr_t) dst % 128 == 0);                                                                       \
    പ്രതിജ്ഞാ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_auu(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    പ്രതിജ്ഞാ((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_uaa(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    പ്രതിജ്ഞാ((uintptr_t) src0 % 128 == 0);                                                                      \
    പ്രതിജ്ഞാ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_uau(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    പ്രതിജ്ഞാ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_uua(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    പ്രതിജ്ഞാ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_uuu(uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_ദ്വിമാന_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, ദശാംശം)
DEFINE_HVX_ദ്വിമാന_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, ദശാംശം)
DEFINE_HVX_ദ്വിമാന_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, ദശാംശം)

DEFINE_HVX_ദ്വിമാന_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_ദ്വിമാന_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_ദ്വിമാന_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#define HVX_ദ്വിമാന_DISPATCHER(OP_NAME)                                                                                                       \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src0, സ്ഥിരം uint8_t * restrict src1, സ്ഥിരം uint32_t അക്കം_elems) { \
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {                                                                                                 \
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src0, 128)) {                                                                                            \
            എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, അക്കം_elems);                                               \
            അല്ലെങ്കിൽ                                    OP_NAME##_aau(dst, src0, src1, അക്കം_elems);                                               \
        } അല്ലെങ്കിൽ {                                                                                                                             \
            എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src1, 128)) OP_NAME##_aua(dst, src0, src1, അക്കം_elems);                                               \
            അല്ലെങ്കിൽ                                    OP_NAME##_auu(dst, src0, src1, അക്കം_elems);                                               \
        }                                                                                                                                    \
    } അല്ലെങ്കിൽ {                                                                                                                                 \
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src0, 128)) {                                                                                            \
            എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, അക്കം_elems);                                               \
            അല്ലെങ്കിൽ                                    OP_NAME##_uau(dst, src0, src1, അക്കം_elems);                                               \
        } അല്ലെങ്കിൽ {                                                                                                                             \
            എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src1, 128)) OP_NAME##_uua(dst, src0, src1, അക്കം_elems);                                               \
            അല്ലെങ്കിൽ                                    OP_NAME##_uuu(dst, src0, src1, അക്കം_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_ദ്വിമാന_DISPATCHER(hvx_add_f32)
HVX_ദ്വിമാന_DISPATCHER(hvx_sub_f32)
HVX_ദ്വിമാന_DISPATCHER(hvx_mul_f32)

HVX_ദ്വിമാന_DISPATCHER(hvx_add_f16)
HVX_ദ്വിമാന_DISPATCHER(hvx_sub_f16)
HVX_ദ്വിമാന_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_mul_mul_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src0, സ്ഥിരം uint8_t * restrict src1, സ്ഥിരം uint8_t * restrict src2, സ്ഥിരം uint32_t അക്കം_elems) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src0 % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src1 % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    സ്ഥിരം uint32_t elem_size = വലുപ്പം(ദശാംശം);
    സ്ഥിരം uint32_t epv  = 128 / elem_size;
    സ്ഥിരം uint32_t nvec = അക്കം_elems / epv;
    സ്ഥിരം uint32_t nloe = അക്കം_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    വേണ്ടി (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    എങ്കിൽ (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((ശൂന്യം *) &vdst[i], nloe * elem_size, v2);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_mul_mul_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src0, സ്ഥിരം uint8_t * restrict src1, സ്ഥിരം uint8_t * restrict src2, സ്ഥിരം uint32_t അക്കം_elems) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src0 % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src1 % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    സ്ഥിരം uint32_t elem_size = വലുപ്പം(_Float16);
    സ്ഥിരം uint32_t epv  = 128 / elem_size;
    സ്ഥിരം uint32_t nvec = അക്കം_elems / epv;
    സ്ഥിരം uint32_t nloe = അക്കം_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    വേണ്ടി (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    എങ്കിൽ (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((ശൂന്യം *) &vdst[i], nloe * elem_size, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, elem_size, vec_store, scalar_op_macro) \
    ചെയ്യുക {                                                                                \
        dst_type * restrict vdst = (dst_type *) dst;                                    \
        src_type * restrict vsrc = (src_type *) src;                                    \
                                                                                        \
        സ്ഥിരം uint32_t epv  = 128 / (elem_size);                                        \
        സ്ഥിരം uint32_t nvec = n / epv;                                                  \
        സ്ഥിരം uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        വേണ്ടി (; i < nvec; i++) {                                                         \
            HVX_Vector v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        എങ്കിൽ (nloe) {                                                                     \
            HVX_Vector v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((ശൂന്യം *) &vdst[i], nloe * (elem_size), v);                        \
        }                                                                               \
    } ആകുമ്പോൾ(0)

#define HVX_OP_ADD_SCALAR_F32(v)                                   \
    ({                                                             \
        സ്ഥിരം HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#define HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#define HVX_OP_ADD_SCALAR_F16(v)                                   \
    ({                                                             \
        സ്ഥിരം HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, out);                         \
    })

#define HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#define HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ELEM_TYPE val, uint32_t n) { \
    സ്ഥിരം HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    സ്ഥിരം HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    പ്രതിജ്ഞാ((uintptr_t) dst % 128 == 0);                                                                                  \
    പ്രതിജ്ഞാ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ELEM_TYPE val, uint32_t n) { \
    സ്ഥിരം HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    സ്ഥിരം HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    പ്രതിജ്ഞാ((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ELEM_TYPE val, uint32_t n) { \
    സ്ഥിരം HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    സ്ഥിരം HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    പ്രതിജ്ഞാ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME##_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ELEM_TYPE val, uint32_t n) { \
    സ്ഥിരം HVX_Vector val_vec = SPLAT_MACRO(val);                                                                         \
    സ്ഥിരം HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, വലുപ്പം(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, ദശാംശം)
DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, ദശാംശം)
DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, ദശാംശം)

DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#define HVX_ദ്വിമാന_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
സ്ഥിതം വരിയിൽ ശൂന്യം OP_NAME(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ELEM_TYPE val, സ്ഥിരം uint32_t അക്കം_elems) { \
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128) && hex_is_aligned((ശൂന്യം *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, അക്കം_elems);                                                                                   \
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, അക്കം_elems);                                                                                   \
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, അക്കം_elems);                                                                                   \
    } അല്ലെങ്കിൽ {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, അക്കം_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_ദ്വിമാന_SCALAR_DISPATCHER(hvx_add_scalar_f32, ദശാംശം)
HVX_ദ്വിമാന_SCALAR_DISPATCHER(hvx_sub_scalar_f32, ദശാംശം)
HVX_ദ്വിമാന_SCALAR_DISPATCHER(hvx_mul_scalar_f32, ദശാംശം)

HVX_ദ്വിമാന_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_ദ്വിമാന_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_ദ്വിമാന_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f32_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f32_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f32(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128) && hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ {
        hvx_min_scalar_f32_uu(dst, src, val, അക്കം_elems);
    }
}

// MAX Scalar variants

#define HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_max_scalar_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_max_scalar_f32_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_max_scalar_f32_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_max_scalar_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_max_scalar_f32(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128) && hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ {
        hvx_max_scalar_f32_uu(dst, src, val, അക്കം_elems);
    }
}

#define HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f16(val);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, വലുപ്പം(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f16_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f16(val);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, വലുപ്പം(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f16_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f16(val);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, വലുപ്പം(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f16_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 val, uint32_t n) {
    സ്ഥിരം HVX_Vector val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, വലുപ്പം(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_min_scalar_f16(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 val, സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128) && hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, അക്കം_elems);
    } അല്ലെങ്കിൽ {
        hvx_min_scalar_f16_uu(dst, src, val, അക്കം_elems);
    }
}

// CLAMP Scalar variants

#define HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_VectorPred pred_cap_വലത് = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_ഇടത്  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_വലത്, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ഇടത്, min_vec, tmp);                      \
    })

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം min, സ്ഥിരം ദശാംശം max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f32(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f32(max);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f32_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം min, സ്ഥിരം ദശാംശം max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f32(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f32(max);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം min, സ്ഥിരം ദശാംശം max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f32(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f32(max);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം min, സ്ഥിരം ദശാംശം max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f32(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f32(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം min, സ്ഥിരം ദശാംശം max, സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128) && hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, അക്കം_elems);
    } അല്ലെങ്കിൽ {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, അക്കം_elems);
    }
}

#define HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_VectorPred pred_cap_വലത് = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_VectorPred pred_cap_ഇടത്  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_വലത്, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ഇടത്, min_vec, tmp);                      \
    })

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 min, സ്ഥിരം _Float16 max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f16(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f16(max);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, വലുപ്പം(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f16_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 min, സ്ഥിരം _Float16 max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f16(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f16(max);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, വലുപ്പം(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f16_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 min, സ്ഥിരം _Float16 max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f16(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f16(max);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, വലുപ്പം(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f16_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 min, സ്ഥിരം _Float16 max, uint32_t n) {
    സ്ഥിരം HVX_Vector min_vec = hvx_vec_splat_f16(min);
    സ്ഥിരം HVX_Vector max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, വലുപ്പം(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_clamp_scalar_f16(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 min, സ്ഥിരം _Float16 max, സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128) && hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, അക്കം_elems);
    } അല്ലെങ്കിൽ {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, അക്കം_elems);
    }
}

#define HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_VectorPred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_Vector     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_leaky_relu_scalar_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം ns, uint32_t n) {
    സ്ഥിരം HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    സ്ഥിരം HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_leaky_relu_scalar_f32_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം ns, uint32_t n) {
    സ്ഥിരം HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    സ്ഥിരം HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_leaky_relu_scalar_f32_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം ns, uint32_t n) {
    സ്ഥിരം HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    സ്ഥിരം HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_leaky_relu_scalar_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം ns, uint32_t n) {
    സ്ഥിരം HVX_Vector zero_vec = hvx_vec_splat_f32(0.0f);
    സ്ഥിരം HVX_Vector ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, വലുപ്പം(ദശാംശം), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_leaky_relu_scalar_f32(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം ns, സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128) && hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, അക്കം_elems);
    } അല്ലെങ്കിൽ എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, അക്കം_elems);
    } അല്ലെങ്കിൽ {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, അക്കം_elems);
    }
}

//
// Abs
//

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_abs_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    സ്ഥിരം uint32_t elem_size = വലുപ്പം(ദശാംശം);
    സ്ഥിരം uint32_t epv       = 128 / elem_size;
    സ്ഥിരം uint32_t nvec      = n / epv;
    സ്ഥിരം uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    വേണ്ടി (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    എങ്കിൽ (nloe) {
        HVX_Vector v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((ശൂന്യം *) &vdst[i], nloe * elem_size, v);
    }
}

#define hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    ചെയ്യുക {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        സ്ഥിരം uint32_t elem_size = വലുപ്പം(_Float16);                       \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                             \
        സ്ഥിരം uint32_t nvec = n / epv;                                     \
        സ്ഥിരം uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        വേണ്ടി (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        എങ്കിൽ (nloe) {                                                        \
            HVX_Vector v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ആകുമ്പോൾ(0)

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_abs_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_abs_f16_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_abs_f16_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_abs_f16_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_abs_f16(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം uint32_t അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
            hvx_abs_f16_aa(dst, src, അക്കം_elems);
        } അല്ലെങ്കിൽ {
            hvx_abs_f16_au(dst, src, അക്കം_elems);
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
            hvx_abs_f16_ua(dst, src, അക്കം_elems);
        } അല്ലെങ്കിൽ {
            hvx_abs_f16_uu(dst, src, അക്കം_elems);
        }
    }
}

//
// Step
//

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_step_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    സ്ഥിരം uint32_t elem_size = വലുപ്പം(ദശാംശം);
    സ്ഥിരം uint32_t epv       = 128 / elem_size;
    സ്ഥിരം uint32_t nvec      = n / epv;
    സ്ഥിരം uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    വേണ്ടി (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f32(vsrc[i]);
    }
    എങ്കിൽ (nloe) {
        HVX_Vector v = hvx_vec_step_f32(vsrc[i]);
        hvx_vec_store_a((ശൂന്യം *) &vdst[i], nloe * elem_size, v);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_step_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    സ്ഥിരം uint32_t elem_size = വലുപ്പം(_Float16);
    സ്ഥിരം uint32_t epv       = 128 / elem_size;
    സ്ഥിരം uint32_t nvec      = n / epv;
    സ്ഥിരം uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    വേണ്ടി (; i < nvec; i++) {
        vdst[i] = hvx_vec_step_f16(vsrc[i]);
    }
    എങ്കിൽ (nloe) {
        HVX_Vector v = hvx_vec_step_f16(vsrc[i]);
        hvx_vec_store_a((ശൂന്യം *) &vdst[i], nloe * elem_size, v);
    }
}

//
// Square
//

#define hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    ചെയ്യുക {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        സ്ഥിരം uint32_t elem_size = വലുപ്പം(ദശാംശം);                          \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                             \
        സ്ഥിരം uint32_t nvec = n / epv;                                     \
        സ്ഥിരം uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        വേണ്ടി (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        എങ്കിൽ (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ആകുമ്പോൾ(0)

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f32_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f32_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f32(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം uint32_t അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, അക്കം_elems);
        } അല്ലെങ്കിൽ {
            hvx_sqr_f32_au(dst, src, അക്കം_elems);
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, അക്കം_elems);
        } അല്ലെങ്കിൽ {
            hvx_sqr_f32_uu(dst, src, അക്കം_elems);
        }
    }
}

#define hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    ചെയ്യുക {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        സ്ഥിരം uint32_t elem_size = വലുപ്പം(_Float16);                       \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                             \
        സ്ഥിരം uint32_t nvec = n / epv;                                     \
        സ്ഥിരം uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        വേണ്ടി (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        എങ്കിൽ (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } ആകുമ്പോൾ(0)

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f16_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f16_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f16_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_sqr_f16(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം uint32_t അക്കം_elems) {
    എങ്കിൽ (hex_is_aligned((ശൂന്യം *) dst, 128)) {
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, അക്കം_elems);
        } അല്ലെങ്കിൽ {
            hvx_sqr_f16_au(dst, src, അക്കം_elems);
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (hex_is_aligned((ശൂന്യം *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, അക്കം_elems);
        } അല്ലെങ്കിൽ {
            hvx_sqr_f16_uu(dst, src, അക്കം_elems);
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
#undef DEFINE_HVX_ദ്വിമാന_OP_VARIANTS
#undef DEFINE_HVX_ദ്വിമാന_SCALAR_OP_VARIANTS
#undef HVX_ദ്വിമാന_DISPATCHER
#undef UNUSED

#endif // HVX_ARITH_H
