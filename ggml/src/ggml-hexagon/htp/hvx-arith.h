#اگر_تعریف_نشده HVX_ARITH_H
#تعریف_کن HVX_ARITH_H

#ضمن <assert.h>
#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <math.h>

#ضمن "hvx-base.h"
#ضمن "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#تعریف_کن استفادهنشده(x) (فراغ)(x)

#تعریف_کن hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_اندازه, vec_store, vec_op) \
    افعل {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        ثابت uint32_t epv  = 128 / (elem_اندازه);                                          \
        ثابت uint32_t nvec = n / epv;                                                    \
        ثابت uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        لكل (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        إذا (nloe) {                                                                       \
            HVX_وکتور v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((فراغ *) &vdst[i], nloe * (elem_اندازه), v);                          \
        }                                                                                 \
    } طالما(0)

#إذا __HVX_ARCH__ < 79

#تعریف_کن HVX_OP_جمع_کن_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#تعریف_کن HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#تعریف_کن HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#وإلا

#تعریف_کن HVX_OP_جمع_کن_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#تعریف_کن HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#تعریف_کن HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#پایان_اگر

#تعریف_کن HVX_OP_جمع_کن_F16(a, b) hvx_vec_جمع_کن_f16_f16(a, b)
#تعریف_کن HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#تعریف_کن HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#تعریف_کن DEFINE_HVX_دودویی_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
ساكن مضمن فراغ OP_NAME##_aaa(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_وکتور, HVX_وکتور, HVX_وکتور, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_aau(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_وکتور, HVX_وکتور, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_aua(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_وکتور, HVX_UVector, HVX_وکتور, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_auu(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_وکتور, HVX_UVector, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uaa(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_وکتور, HVX_وکتور, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uau(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_وکتور, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uua(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_وکتور, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uuu(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_دودویی_OP_VARIANTS(hvx_جمع_کن_f32, HVX_OP_جمع_کن_F32, عائم)
DEFINE_HVX_دودویی_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, عائم)
DEFINE_HVX_دودویی_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, عائم)

DEFINE_HVX_دودویی_OP_VARIANTS(hvx_جمع_کن_f16, HVX_OP_جمع_کن_F16, _Float16)
DEFINE_HVX_دودویی_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_دودویی_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#تعریف_کن HVX_دودویی_DISPATCHER(OP_NAME)                                                                                                       \
ساكن مضمن فراغ OP_NAME(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint32_t رقم_elems) { \
    إذا (hex_is_aligned((فراغ *) dst, 128)) {                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, رقم_elems);                                               \
            وإلا                                    OP_NAME##_aau(dst, src0, src1, رقم_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, رقم_elems);                                               \
            وإلا                                    OP_NAME##_auu(dst, src0, src1, رقم_elems);                                               \
        }                                                                                                                                    \
    } وإلا {                                                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, رقم_elems);                                               \
            وإلا                                    OP_NAME##_uau(dst, src0, src1, رقم_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, رقم_elems);                                               \
            وإلا                                    OP_NAME##_uuu(dst, src0, src1, رقم_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_دودویی_DISPATCHER(hvx_جمع_کن_f32)
HVX_دودویی_DISPATCHER(hvx_sub_f32)
HVX_دودویی_DISPATCHER(hvx_mul_f32)

HVX_دودویی_DISPATCHER(hvx_جمع_کن_f16)
HVX_دودویی_DISPATCHER(hvx_sub_f16)
HVX_دودویی_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
ساكن مضمن فراغ hvx_mul_mul_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint8_t * محدود src2, ثابت uint32_t رقم_elems) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src0 % 128 == 0);
    أكد((بیعلامت دراز) src1 % 128 == 0);
    أكد((بیعلامت دراز) src2 % 128 == 0);

    HVX_وکتور * محدود vdst  = (HVX_وکتور *) dst;
    HVX_وکتور * محدود vsrc0 = (HVX_وکتور *) src0;
    HVX_وکتور * محدود vsrc1 = (HVX_وکتور *) src1;
    HVX_وکتور * محدود vsrc2 = (HVX_وکتور *) src2;

    ثابت uint32_t elem_اندازه = حجم(عائم);
    ثابت uint32_t epv  = 128 / elem_اندازه;
    ثابت uint32_t nvec = رقم_elems / epv;
    ثابت uint32_t nloe = رقم_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_وکتور v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    إذا (nloe) {
        HVX_وکتور v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_وکتور v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v2);
    }
}

ساكن مضمن فراغ hvx_mul_mul_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint8_t * محدود src2, ثابت uint32_t رقم_elems) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src0 % 128 == 0);
    أكد((بیعلامت دراز) src1 % 128 == 0);
    أكد((بیعلامت دراز) src2 % 128 == 0);

    HVX_وکتور * محدود vdst  = (HVX_وکتور *) dst;
    HVX_وکتور * محدود vsrc0 = (HVX_وکتور *) src0;
    HVX_وکتور * محدود vsrc1 = (HVX_وکتور *) src1;
    HVX_وکتور * محدود vsrc2 = (HVX_وکتور *) src2;

    ثابت uint32_t elem_اندازه = حجم(_Float16);
    ثابت uint32_t epv  = 128 / elem_اندازه;
    ثابت uint32_t nvec = رقم_elems / epv;
    ثابت uint32_t nloe = رقم_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_وکتور v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    إذا (nloe) {
        HVX_وکتور v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_وکتور v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v2);
    }
}

// Scalar Operations

#تعریف_کن hvx_scalar_loop_body(dst_type, src_type, elem_اندازه, vec_store, scalar_op_macro) \
    افعل {                                                                                \
        dst_type * محدود vdst = (dst_type *) dst;                                    \
        src_type * محدود vsrc = (src_type *) src;                                    \
                                                                                        \
        ثابت uint32_t epv  = 128 / (elem_اندازه);                                        \
        ثابت uint32_t nvec = n / epv;                                                  \
        ثابت uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        لكل (; i < nvec; i++) {                                                         \
            HVX_وکتور v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        إذا (nloe) {                                                                     \
            HVX_وکتور v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((فراغ *) &vdst[i], nloe * (elem_اندازه), v);                        \
        }                                                                               \
    } طالما(0)

#تعریف_کن HVX_OP_جمع_کن_SCALAR_F32(v)                                   \
    ({                                                             \
        ثابت HVX_وکتور_Pred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_وکتور بیرون = HVX_OP_جمع_کن_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, بیرون);                         \
    })

#تعریف_کن HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#تعریف_کن HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#تعریف_کن HVX_OP_جمع_کن_SCALAR_F16(v)                                   \
    ({                                                             \
        ثابت HVX_وکتور_Pred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_وکتور بیرون = HVX_OP_جمع_کن_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, بیرون);                         \
    })

#تعریف_کن HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#تعریف_کن HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#تعریف_کن DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
ساكن مضمن فراغ OP_NAME##_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_وکتور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_وکتور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); استفادهنشده(inf);                                                \
    أكد((uintptr_t) dst % 128 == 0);                                                                                  \
    أكد((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_وکتور, HVX_وکتور, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
ساكن مضمن فراغ OP_NAME##_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_وکتور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_وکتور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); استفادهنشده(inf);                                                \
    أكد((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_وکتور, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
ساكن مضمن فراغ OP_NAME##_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_وکتور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_وکتور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); استفادهنشده(inf);                                                \
    أكد((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_وکتور, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
ساكن مضمن فراغ OP_NAME##_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_وکتور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_وکتور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); استفادهنشده(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS(hvx_جمع_کن_scalar_f32, HVX_OP_جمع_کن_SCALAR_F32, hvx_vec_splat_f32, عائم)
DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, عائم)
DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, عائم)

DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS(hvx_جمع_کن_scalar_f16, HVX_OP_جمع_کن_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#تعریف_کن HVX_دودویی_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
ساكن مضمن فراغ OP_NAME(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, ثابت uint32_t رقم_elems) { \
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, رقم_elems);                                                                                   \
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, رقم_elems);                                                                                   \
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, رقم_elems);                                                                                   \
    } وإلا {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, رقم_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_دودویی_SCALAR_DISPATCHER(hvx_جمع_کن_scalar_f32, عائم)
HVX_دودویی_SCALAR_DISPATCHER(hvx_sub_scalar_f32, عائم)
HVX_دودویی_SCALAR_DISPATCHER(hvx_mul_scalar_f32, عائم)

HVX_دودویی_SCALAR_DISPATCHER(hvx_جمع_کن_scalar_f16, _Float16)
HVX_دودویی_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_دودویی_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#تعریف_کن HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

ساكن مضمن فراغ hvx_min_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_وکتور, حجم(عائم), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_وکتور, حجم(عائم), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, رقم_elems);
    } وإلا {
        hvx_min_scalar_f32_uu(dst, src, val, رقم_elems);
    }
}

// MAX Scalar variants

#تعریف_کن HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

ساكن مضمن فراغ hvx_max_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_وکتور, حجم(عائم), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_وکتور, حجم(عائم), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, رقم_elems);
    } وإلا {
        hvx_max_scalar_f32_uu(dst, src, val, رقم_elems);
    }
}

#تعریف_کن HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

ساكن مضمن فراغ hvx_min_scalar_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f16(val);
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_وکتور, حجم(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f16(val);
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_UVector, حجم(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f16(val);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_وکتور, حجم(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_وکتور val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, رقم_elems);
    } وإلا {
        hvx_min_scalar_f16_uu(dst, src, val, رقم_elems);
    }
}

// CLAMP Scalar variants

#تعریف_کن HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_وکتور_Pred pred_cap_راست = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_وکتور_Pred pred_cap_چپ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_وکتور tmp = Q6_V_vmux_QVV(pred_cap_راست, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_چپ, min_vec, tmp);                      \
    })

ساكن مضمن فراغ hvx_clamp_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f32(max);
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_وکتور, حجم(عائم), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f32(max);
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f32(max);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_وکتور, حجم(عائم), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, رقم_elems);
    } وإلا {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, رقم_elems);
    }
}

#تعریف_کن HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_وکتور_Pred pred_cap_راست = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_وکتور_Pred pred_cap_چپ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_وکتور tmp = Q6_V_vmux_QVV(pred_cap_راست, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_چپ, min_vec, tmp);                      \
    })

ساكن مضمن فراغ hvx_clamp_scalar_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f16(max);
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_وکتور, حجم(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f16(max);
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_UVector, حجم(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f16(max);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_وکتور, حجم(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_وکتور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_وکتور max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, رقم_elems);
    } وإلا {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, رقم_elems);
    }
}

#تعریف_کن HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_وکتور_Pred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_وکتور     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_وکتور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_وکتور ns_vec   = hvx_vec_splat_f32(ns);
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_وکتور, حجم(عائم), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_وکتور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_وکتور ns_vec   = hvx_vec_splat_f32(ns);
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_وکتور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_وکتور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_وکتور ns_vec   = hvx_vec_splat_f32(ns);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_وکتور, حجم(عائم), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_وکتور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_وکتور ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, ثابت صحيح رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, رقم_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, رقم_elems);
    } وإلا {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, رقم_elems);
    }
}

//
// Abs
//

ساكن مضمن فراغ hvx_abs_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);

    HVX_وکتور * محدود vdst = (HVX_وکتور *) dst;
    HVX_وکتور * محدود vsrc = (HVX_وکتور *) src;

    ثابت uint32_t elem_اندازه = حجم(عائم);
    ثابت uint32_t epv       = 128 / elem_اندازه;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    إذا (nloe) {
        HVX_وکتور v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

#تعریف_کن hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    افعل {                                                                   \
        dst_type * محدود vdst  = (dst_type *) dst;                      \
        src_type * محدود vsrc = (src_type *) src;                       \
                                                                           \
        ثابت uint32_t elem_اندازه = حجم(_Float16);                       \
        ثابت uint32_t epv  = 128 / elem_اندازه;                             \
        ثابت uint32_t nvec = n / epv;                                     \
        ثابت uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        لكل (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        إذا (nloe) {                                                        \
            HVX_وکتور v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((فراغ *) &vdst[i], nloe * elem_اندازه, v);             \
        }                                                                  \
    } طالما(0)

ساكن مضمن فراغ hvx_abs_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_وکتور, HVX_وکتور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_abs_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_وکتور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_abs_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_وکتور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_abs_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_abs_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128)) {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_abs_f16_aa(dst, src, رقم_elems);
        } وإلا {
            hvx_abs_f16_au(dst, src, رقم_elems);
        }
    } وإلا {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_abs_f16_ua(dst, src, رقم_elems);
        } وإلا {
            hvx_abs_f16_uu(dst, src, رقم_elems);
        }
    }
}

//
//گام
//

ساكن مضمن فراغ hvx_گام_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);

    HVX_وکتور * محدود vdst = (HVX_وکتور *) dst;
    HVX_وکتور * محدود vsrc = (HVX_وکتور *) src;

    ثابت uint32_t elem_اندازه = حجم(عائم);
    ثابت uint32_t epv       = 128 / elem_اندازه;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_گام_f32(vsrc[i]);
    }
    إذا (nloe) {
        HVX_وکتور v = hvx_vec_گام_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

ساكن مضمن فراغ hvx_گام_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);

    HVX_وکتور * محدود vdst = (HVX_وکتور *) dst;
    HVX_وکتور * محدود vsrc = (HVX_وکتور *) src;

    ثابت uint32_t elem_اندازه = حجم(_Float16);
    ثابت uint32_t epv       = 128 / elem_اندازه;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_گام_f16(vsrc[i]);
    }
    إذا (nloe) {
        HVX_وکتور v = hvx_vec_گام_f16(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

//
//مربع
//

#تعریف_کن hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    افعل {                                                                   \
        dst_type * محدود vdst  = (dst_type *) dst;                      \
        src_type * محدود vsrc = (src_type *) src;                       \
                                                                           \
        ثابت uint32_t elem_اندازه = حجم(عائم);                          \
        ثابت uint32_t epv  = 128 / elem_اندازه;                             \
        ثابت uint32_t nvec = n / epv;                                     \
        ثابت uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        لكل (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        إذا (nloe) {                                                        \
            HVX_وکتور v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((فراغ *) &vdst[i], nloe * elem_اندازه, v);             \
        }                                                                  \
    } طالما(0)

ساكن مضمن فراغ hvx_sqr_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_وکتور, HVX_وکتور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_وکتور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_وکتور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128)) {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, رقم_elems);
        } وإلا {
            hvx_sqr_f32_au(dst, src, رقم_elems);
        }
    } وإلا {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, رقم_elems);
        } وإلا {
            hvx_sqr_f32_uu(dst, src, رقم_elems);
        }
    }
}

#تعریف_کن hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    افعل {                                                                   \
        dst_type * محدود vdst  = (dst_type *) dst;                      \
        src_type * محدود vsrc = (src_type *) src;                       \
                                                                           \
        ثابت uint32_t elem_اندازه = حجم(_Float16);                       \
        ثابت uint32_t epv  = 128 / elem_اندازه;                             \
        ثابت uint32_t nvec = n / epv;                                     \
        ثابت uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        لكل (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        إذا (nloe) {                                                        \
            HVX_وکتور v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((فراغ *) &vdst[i], nloe * elem_اندازه, v);             \
        }                                                                  \
    } طالما(0)

ساكن مضمن فراغ hvx_sqr_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_وکتور, HVX_وکتور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_وکتور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_وکتور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t رقم_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128)) {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, رقم_elems);
        } وإلا {
            hvx_sqr_f16_au(dst, src, رقم_elems);
        }
    } وإلا {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, رقم_elems);
        } وإلا {
            hvx_sqr_f16_uu(dst, src, رقم_elems);
        }
    }
}

#لغو_تعریف HVX_OP_جمع_کن_F32
#لغو_تعریف HVX_OP_SUB_F32
#لغو_تعریف HVX_OP_MUL_F32
#لغو_تعریف HVX_OP_جمع_کن_F16
#لغو_تعریف HVX_OP_SUB_F16
#لغو_تعریف HVX_OP_MUL_F16
#لغو_تعریف hvx_arith_loop_body
#لغو_تعریف HVX_OP_جمع_کن_SCALAR_F32
#لغو_تعریف HVX_OP_SUB_SCALAR_F32
#لغو_تعریف HVX_OP_MUL_SCALAR_F32
#لغو_تعریف HVX_OP_جمع_کن_SCALAR_F16
#لغو_تعریف HVX_OP_SUB_SCALAR_F16
#لغو_تعریف HVX_OP_MUL_SCALAR_F16
#لغو_تعریف hvx_scalar_loop_body
#لغو_تعریف HVX_OP_MIN_SCALAR
#لغو_تعریف HVX_OP_MAX_SCALAR
#لغو_تعریف HVX_OP_MIN_SCALAR_F16
#لغو_تعریف HVX_OP_CLAMP_SCALAR
#لغو_تعریف HVX_OP_CLAMP_SCALAR_F16
#لغو_تعریف HVX_OP_LEAKY_RELU_SCALAR
#لغو_تعریف DEFINE_HVX_دودویی_OP_VARIANTS
#لغو_تعریف DEFINE_HVX_دودویی_SCALAR_OP_VARIANTS
#لغو_تعریف HVX_دودویی_DISPATCHER
#لغو_تعریف استفادهنشده

#پایان_اگر // HVX_ARITH_H
