#که_نه_تعریف_شوی HVX_ARITH_H
#تعریف_کړه HVX_ARITH_H

#ضمن <assert.h>
#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <math.h>

#ضمن "hvx-base.h"
#ضمن "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#تعریف_کړه ناکارول_شوی(x) (فراغ)(x)

#تعریف_کړه hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_اندازه, vec_store, vec_op) \
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
            HVX_ویکټور v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((فراغ *) &vdst[i], nloe * (elem_اندازه), v);                          \
        }                                                                                 \
    } طالما(0)

#إذا __HVX_ARCH__ < 79

#تعریف_کړه HVX_OP_جمع_کړه_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#تعریف_کړه HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#تعریف_کړه HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#وإلا

#تعریف_کړه HVX_OP_جمع_کړه_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#تعریف_کړه HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#تعریف_کړه HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#د_که_پای

#تعریف_کړه HVX_OP_جمع_کړه_F16(a, b) hvx_vec_جمع_کړه_f16_f16(a, b)
#تعریف_کړه HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#تعریف_کړه HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#تعریف_کړه DEFINE_HVX_دوګونی_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
ساكن مضمن فراغ OP_NAME##_aaa(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ویکټور, HVX_ویکټور, HVX_ویکټور, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_aau(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ویکټور, HVX_ویکټور, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_aua(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ویکټور, HVX_UVector, HVX_ویکټور, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_auu(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_ویکټور, HVX_UVector, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uaa(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_ویکټور, HVX_ویکټور, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uau(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_ویکټور, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uua(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    أكد((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_ویکټور, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ساكن مضمن فراغ OP_NAME##_uuu(uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_دوګونی_OP_VARIANTS(hvx_جمع_کړه_f32, HVX_OP_جمع_کړه_F32, عائم)
DEFINE_HVX_دوګونی_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, عائم)
DEFINE_HVX_دوګونی_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, عائم)

DEFINE_HVX_دوګونی_OP_VARIANTS(hvx_جمع_کړه_f16, HVX_OP_جمع_کړه_F16, _Float16)
DEFINE_HVX_دوګونی_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_دوګونی_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#تعریف_کړه HVX_دوګونی_DISPATCHER(OP_NAME)                                                                                                       \
ساكن مضمن فراغ OP_NAME(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint32_t عدد_elems) { \
    إذا (hex_is_aligned((فراغ *) dst, 128)) {                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_aau(dst, src0, src1, عدد_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_auu(dst, src0, src1, عدد_elems);                                               \
        }                                                                                                                                    \
    } وإلا {                                                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_uau(dst, src0, src1, عدد_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_uuu(dst, src0, src1, عدد_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_دوګونی_DISPATCHER(hvx_جمع_کړه_f32)
HVX_دوګونی_DISPATCHER(hvx_sub_f32)
HVX_دوګونی_DISPATCHER(hvx_mul_f32)

HVX_دوګونی_DISPATCHER(hvx_جمع_کړه_f16)
HVX_دوګونی_DISPATCHER(hvx_sub_f16)
HVX_دوګونی_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
ساكن مضمن فراغ hvx_mul_mul_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint8_t * محدود src2, ثابت uint32_t عدد_elems) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src0 % 128 == 0);
    أكد((بینښه اوږد) src1 % 128 == 0);
    أكد((بینښه اوږد) src2 % 128 == 0);

    HVX_ویکټور * محدود vdst  = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc0 = (HVX_ویکټور *) src0;
    HVX_ویکټور * محدود vsrc1 = (HVX_ویکټور *) src1;
    HVX_ویکټور * محدود vsrc2 = (HVX_ویکټور *) src2;

    ثابت uint32_t elem_اندازه = حجم(عائم);
    ثابت uint32_t epv  = 128 / elem_اندازه;
    ثابت uint32_t nvec = عدد_elems / epv;
    ثابت uint32_t nloe = عدد_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ویکټور v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    إذا (nloe) {
        HVX_ویکټور v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_ویکټور v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v2);
    }
}

ساكن مضمن فراغ hvx_mul_mul_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint8_t * محدود src2, ثابت uint32_t عدد_elems) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src0 % 128 == 0);
    أكد((بینښه اوږد) src1 % 128 == 0);
    أكد((بینښه اوږد) src2 % 128 == 0);

    HVX_ویکټور * محدود vdst  = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc0 = (HVX_ویکټور *) src0;
    HVX_ویکټور * محدود vsrc1 = (HVX_ویکټور *) src1;
    HVX_ویکټور * محدود vsrc2 = (HVX_ویکټور *) src2;

    ثابت uint32_t elem_اندازه = حجم(_Float16);
    ثابت uint32_t epv  = 128 / elem_اندازه;
    ثابت uint32_t nvec = عدد_elems / epv;
    ثابت uint32_t nloe = عدد_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ویکټور v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    إذا (nloe) {
        HVX_ویکټور v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_ویکټور v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v2);
    }
}

// Scalar Operations

#تعریف_کړه hvx_scalar_loop_body(dst_type, src_type, elem_اندازه, vec_store, scalar_op_macro) \
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
            HVX_ویکټور v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        إذا (nloe) {                                                                     \
            HVX_ویکټور v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((فراغ *) &vdst[i], nloe * (elem_اندازه), v);                        \
        }                                                                               \
    } طالما(0)

#تعریف_کړه HVX_OP_جمع_کړه_SCALAR_F32(v)                                   \
    ({                                                             \
        ثابت HVX_ویکټور_Pred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_ویکټور بهر = HVX_OP_جمع_کړه_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, بهر);                         \
    })

#تعریف_کړه HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#تعریف_کړه HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#تعریف_کړه HVX_OP_جمع_کړه_SCALAR_F16(v)                                   \
    ({                                                             \
        ثابت HVX_ویکټور_Pred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_ویکټور بهر = HVX_OP_جمع_کړه_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, بهر);                         \
    })

#تعریف_کړه HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#تعریف_کړه HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#تعریف_کړه DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
ساكن مضمن فراغ OP_NAME##_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_ویکټور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_ویکټور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ناکارول_شوی(inf);                                                \
    أكد((uintptr_t) dst % 128 == 0);                                                                                  \
    أكد((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_ویکټور, HVX_ویکټور, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
ساكن مضمن فراغ OP_NAME##_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_ویکټور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_ویکټور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ناکارول_شوی(inf);                                                \
    أكد((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_ویکټور, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
ساكن مضمن فراغ OP_NAME##_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_ویکټور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_ویکټور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ناکارول_شوی(inf);                                                \
    أكد((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_ویکټور, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
ساكن مضمن فراغ OP_NAME##_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, uint32_t n) { \
    ثابت HVX_ویکټور val_vec = SPLAT_MACRO(val);                                                                         \
    ثابت HVX_ویکټور inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ناکارول_شوی(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS(hvx_جمع_کړه_scalar_f32, HVX_OP_جمع_کړه_SCALAR_F32, hvx_vec_splat_f32, عائم)
DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, عائم)
DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, عائم)

DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS(hvx_جمع_کړه_scalar_f16, HVX_OP_جمع_کړه_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#تعریف_کړه HVX_دوګونی_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
ساكن مضمن فراغ OP_NAME(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت ELEM_TYPE val, ثابت uint32_t عدد_elems) { \
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, عدد_elems);                                                                                   \
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, عدد_elems);                                                                                   \
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, عدد_elems);                                                                                   \
    } وإلا {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, عدد_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_دوګونی_SCALAR_DISPATCHER(hvx_جمع_کړه_scalar_f32, عائم)
HVX_دوګونی_SCALAR_DISPATCHER(hvx_sub_scalar_f32, عائم)
HVX_دوګونی_SCALAR_DISPATCHER(hvx_mul_scalar_f32, عائم)

HVX_دوګونی_SCALAR_DISPATCHER(hvx_جمع_کړه_scalar_f16, _Float16)
HVX_دوګونی_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_دوګونی_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#تعریف_کړه HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

ساكن مضمن فراغ hvx_min_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_ویکټور, حجم(عائم), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ویکټور, حجم(عائم), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ساكن مضمن فراغ hvx_min_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, عدد_elems);
    } وإلا {
        hvx_min_scalar_f32_uu(dst, src, val, عدد_elems);
    }
}

// MAX Scalar variants

#تعریف_کړه HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

ساكن مضمن فراغ hvx_max_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_ویکټور, حجم(عائم), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ویکټور, حجم(عائم), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ساكن مضمن فراغ hvx_max_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, عدد_elems);
    } وإلا {
        hvx_max_scalar_f32_uu(dst, src, val, عدد_elems);
    }
}

#تعریف_کړه HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

ساكن مضمن فراغ hvx_min_scalar_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f16(val);
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_ویکټور, حجم(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f16(val);
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_UVector, حجم(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f16(val);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ویکټور, حجم(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ساكن مضمن فراغ hvx_min_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, عدد_elems);
    } وإلا {
        hvx_min_scalar_f16_uu(dst, src, val, عدد_elems);
    }
}

// CLAMP Scalar variants

#تعریف_کړه HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_ویکټور_Pred pred_cap_ښی = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_ویکټور_Pred pred_cap_کیڼ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_ویکټور tmp = Q6_V_vmux_QVV(pred_cap_ښی, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_کیڼ, min_vec, tmp);                      \
    })

ساكن مضمن فراغ hvx_clamp_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f32(max);
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_ویکټور, حجم(عائم), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f32(max);
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f32(max);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ویکټور, حجم(عائم), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f32(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ساكن مضمن فراغ hvx_clamp_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم min, ثابت عائم max, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, عدد_elems);
    } وإلا {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, عدد_elems);
    }
}

#تعریف_کړه HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_ویکټور_Pred pred_cap_ښی = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_ویکټور_Pred pred_cap_کیڼ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_ویکټور tmp = Q6_V_vmux_QVV(pred_cap_ښی, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_کیڼ, min_vec, tmp);                      \
    })

ساكن مضمن فراغ hvx_clamp_scalar_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f16(max);
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_ویکټور, حجم(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f16(max);
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_UVector, حجم(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f16(max);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ویکټور, حجم(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, uint32_t n) {
    ثابت HVX_ویکټور min_vec = hvx_vec_splat_f16(min);
    ثابت HVX_ویکټور max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ساكن مضمن فراغ hvx_clamp_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 min, ثابت _Float16 max, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, عدد_elems);
    } وإلا {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, عدد_elems);
    }
}

#تعریف_کړه HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_ویکټور_Pred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_ویکټور     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_ویکټور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_ویکټور ns_vec   = hvx_vec_splat_f32(ns);
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_ویکټور, حجم(عائم), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_ویکټور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_ویکټور ns_vec   = hvx_vec_splat_f32(ns);
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ویکټور, HVX_UVector, حجم(عائم), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_ویکټور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_ویکټور ns_vec   = hvx_vec_splat_f32(ns);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ویکټور, حجم(عائم), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, uint32_t n) {
    ثابت HVX_ویکټور zero_vec = hvx_vec_splat_f32(0.0f);
    ثابت HVX_ویکټور ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, حجم(عائم), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ساكن مضمن فراغ hvx_leaky_relu_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم ns, ثابت صحيح عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128) && hex_is_aligned((فراغ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, عدد_elems);
    } وإلا إذا (hex_is_aligned((فراغ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, عدد_elems);
    } وإلا {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, عدد_elems);
    }
}

//
// Abs
//

ساكن مضمن فراغ hvx_abs_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);

    HVX_ویکټور * محدود vdst = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc = (HVX_ویکټور *) src;

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
        HVX_ویکټور v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

#تعریف_کړه hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
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
            HVX_ویکټور v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((فراغ *) &vdst[i], nloe * elem_اندازه, v);             \
        }                                                                  \
    } طالما(0)

ساكن مضمن فراغ hvx_abs_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_ویکټور, HVX_ویکټور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_abs_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_ویکټور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_abs_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_ویکټور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_abs_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_abs_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128)) {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_abs_f16_aa(dst, src, عدد_elems);
        } وإلا {
            hvx_abs_f16_au(dst, src, عدد_elems);
        }
    } وإلا {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_abs_f16_ua(dst, src, عدد_elems);
        } وإلا {
            hvx_abs_f16_uu(dst, src, عدد_elems);
        }
    }
}

//
//ګام
//

ساكن مضمن فراغ hvx_ګام_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);

    HVX_ویکټور * محدود vdst = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc = (HVX_ویکټور *) src;

    ثابت uint32_t elem_اندازه = حجم(عائم);
    ثابت uint32_t epv       = 128 / elem_اندازه;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_ګام_f32(vsrc[i]);
    }
    إذا (nloe) {
        HVX_ویکټور v = hvx_vec_ګام_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

ساكن مضمن فراغ hvx_ګام_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);

    HVX_ویکټور * محدود vdst = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc = (HVX_ویکټور *) src;

    ثابت uint32_t elem_اندازه = حجم(_Float16);
    ثابت uint32_t epv       = 128 / elem_اندازه;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_ګام_f16(vsrc[i]);
    }
    إذا (nloe) {
        HVX_ویکټور v = hvx_vec_ګام_f16(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

//
//مربع
//

#تعریف_کړه hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
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
            HVX_ویکټور v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((فراغ *) &vdst[i], nloe * elem_اندازه, v);             \
        }                                                                  \
    } طالما(0)

ساكن مضمن فراغ hvx_sqr_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_ویکټور, HVX_ویکټور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_ویکټور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_ویکټور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128)) {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, عدد_elems);
        } وإلا {
            hvx_sqr_f32_au(dst, src, عدد_elems);
        }
    } وإلا {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, عدد_elems);
        } وإلا {
            hvx_sqr_f32_uu(dst, src, عدد_elems);
        }
    }
}

#تعریف_کړه hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
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
            HVX_ویکټور v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((فراغ *) &vdst[i], nloe * elem_اندازه, v);             \
        }                                                                  \
    } طالما(0)

ساكن مضمن فراغ hvx_sqr_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_ویکټور, HVX_ویکټور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_ویکټور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqr_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_ویکټور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqr_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t عدد_elems) {
    إذا (hex_is_aligned((فراغ *) dst, 128)) {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, عدد_elems);
        } وإلا {
            hvx_sqr_f16_au(dst, src, عدد_elems);
        }
    } وإلا {
        إذا (hex_is_aligned((فراغ *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, عدد_elems);
        } وإلا {
            hvx_sqr_f16_uu(dst, src, عدد_elems);
        }
    }
}

#تعریف_لغوه HVX_OP_جمع_کړه_F32
#تعریف_لغوه HVX_OP_SUB_F32
#تعریف_لغوه HVX_OP_MUL_F32
#تعریف_لغوه HVX_OP_جمع_کړه_F16
#تعریف_لغوه HVX_OP_SUB_F16
#تعریف_لغوه HVX_OP_MUL_F16
#تعریف_لغوه hvx_arith_loop_body
#تعریف_لغوه HVX_OP_جمع_کړه_SCALAR_F32
#تعریف_لغوه HVX_OP_SUB_SCALAR_F32
#تعریف_لغوه HVX_OP_MUL_SCALAR_F32
#تعریف_لغوه HVX_OP_جمع_کړه_SCALAR_F16
#تعریف_لغوه HVX_OP_SUB_SCALAR_F16
#تعریف_لغوه HVX_OP_MUL_SCALAR_F16
#تعریف_لغوه hvx_scalar_loop_body
#تعریف_لغوه HVX_OP_MIN_SCALAR
#تعریف_لغوه HVX_OP_MAX_SCALAR
#تعریف_لغوه HVX_OP_MIN_SCALAR_F16
#تعریف_لغوه HVX_OP_CLAMP_SCALAR
#تعریف_لغوه HVX_OP_CLAMP_SCALAR_F16
#تعریف_لغوه HVX_OP_LEAKY_RELU_SCALAR
#تعریف_لغوه DEFINE_HVX_دوګونی_OP_VARIANTS
#تعریف_لغوه DEFINE_HVX_دوګونی_SCALAR_OP_VARIANTS
#تعریف_لغوه HVX_دوګونی_DISPATCHER
#تعریف_لغوه ناکارول_شوی

#د_که_پای // HVX_ARITH_H
