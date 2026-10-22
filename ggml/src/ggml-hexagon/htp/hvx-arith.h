#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_ARITH_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_ARITH_H

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ(x) (ਖਾਲੀ)(x)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_ਮਾਪ, vec_store, vec_op) \
    ਕਰੋ {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        ਅਚਲ uint32_t epv  = 128 / (elem_ਮਾਪ);                                          \
        ਅਚਲ uint32_t nvec = n / epv;                                                    \
        ਅਚਲ uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ਲਈ (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        ਜੇ (nloe) {                                                                       \
            HVX_ਸਦਿਸ਼ v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * (elem_ਮਾਪ), v);                          \
        }                                                                                 \
    } ਜਦੋਂ(0)

#ਜੇ __HVX_ARCH__ < 79

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#ਨਹੀਂਤਾਂ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_F16(a, b) hvx_vec_ਜੋੜੋ_f16_f16(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_aaa(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                       \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                      \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_aau(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                       \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_aua(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                       \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_auu(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, HVX_UVector, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uaa(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                      \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uau(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uua(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uuu(uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS(hvx_ਜੋੜੋ_f32, HVX_OP_ਜੋੜੋ_F32, ਦਸ਼ਮਲਵ)
DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, ਦਸ਼ਮਲਵ)
DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, ਦਸ਼ਮਲਵ)

DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS(hvx_ਜੋੜੋ_f16, HVX_OP_ਜੋੜੋ_F16, _Float16)
DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_ਦੁਆਧਾਰੀ_DISPATCHER(OP_NAME)                                                                                                       \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, ਅਚਲ uint32_t ਅੰਕ_elems) { \
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {                                                                                                 \
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src0, 128)) {                                                                                            \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_aau(dst, src0, src1, ਅੰਕ_elems);                                               \
        } ਨਹੀਂਤਾਂ {                                                                                                                             \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_auu(dst, src0, src1, ਅੰਕ_elems);                                               \
        }                                                                                                                                    \
    } ਨਹੀਂਤਾਂ {                                                                                                                                 \
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src0, 128)) {                                                                                            \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_uau(dst, src0, src1, ਅੰਕ_elems);                                               \
        } ਨਹੀਂਤਾਂ {                                                                                                                             \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_uuu(dst, src0, src1, ਅੰਕ_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_ਦੁਆਧਾਰੀ_DISPATCHER(hvx_ਜੋੜੋ_f32)
HVX_ਦੁਆਧਾਰੀ_DISPATCHER(hvx_sub_f32)
HVX_ਦੁਆਧਾਰੀ_DISPATCHER(hvx_mul_f32)

HVX_ਦੁਆਧਾਰੀ_DISPATCHER(hvx_ਜੋੜੋ_f16)
HVX_ਦੁਆਧਾਰੀ_DISPATCHER(hvx_sub_f16)
HVX_ਦੁਆਧਾਰੀ_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_mul_mul_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, ਅਚਲ uint8_t * ਸੀਮਿਤ src2, ਅਚਲ uint32_t ਅੰਕ_elems) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src0 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src1 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src2 % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst  = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc0 = (HVX_ਸਦਿਸ਼ *) src0;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc1 = (HVX_ਸਦਿਸ਼ *) src1;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc2 = (HVX_ਸਦਿਸ਼ *) src2;

    ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t nvec = ਅੰਕ_elems / epv;
    ਅਚਲ uint32_t nloe = ਅੰਕ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_ਸਦਿਸ਼ v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v2);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_mul_mul_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, ਅਚਲ uint8_t * ਸੀਮਿਤ src2, ਅਚਲ uint32_t ਅੰਕ_elems) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src0 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src1 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src2 % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst  = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc0 = (HVX_ਸਦਿਸ਼ *) src0;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc1 = (HVX_ਸਦਿਸ਼ *) src1;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc2 = (HVX_ਸਦਿਸ਼ *) src2;

    ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(_Float16);
    ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t nvec = ਅੰਕ_elems / epv;
    ਅਚਲ uint32_t nloe = ਅੰਕ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_ਸਦਿਸ਼ v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v2);
    }
}

// Scalar Operations

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_scalar_loop_body(dst_type, src_type, elem_ਮਾਪ, vec_store, scalar_op_macro) \
    ਕਰੋ {                                                                                \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                    \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                    \
                                                                                        \
        ਅਚਲ uint32_t epv  = 128 / (elem_ਮਾਪ);                                        \
        ਅਚਲ uint32_t nvec = n / epv;                                                  \
        ਅਚਲ uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        ਲਈ (; i < nvec; i++) {                                                         \
            HVX_ਸਦਿਸ਼ v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        ਜੇ (nloe) {                                                                     \
            HVX_ਸਦਿਸ਼ v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * (elem_ਮਾਪ), v);                        \
        }                                                                               \
    } ਜਦੋਂ(0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_SCALAR_F32(v)                                   \
    ({                                                             \
        ਅਚਲ HVX_ਸਦਿਸ਼_Pred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ = HVX_OP_ਜੋੜੋ_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, ਬਾਹਰਲਾ);                         \
    })

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_SCALAR_F16(v)                                   \
    ({                                                             \
        ਅਚਲ HVX_ਸਦਿਸ਼_Pred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ = HVX_OP_ਜੋੜੋ_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, ਬਾਹਰਲਾ);                         \
    })

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ELEM_TYPE val, uint32_t n) { \
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = SPLAT_MACRO(val);                                                                         \
    ਅਚਲ HVX_ਸਦਿਸ਼ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ਅਣਵਰਤਿਆ(inf);                                                \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                                  \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ELEM_TYPE val, uint32_t n) { \
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = SPLAT_MACRO(val);                                                                         \
    ਅਚਲ HVX_ਸਦਿਸ਼ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ਅਣਵਰਤਿਆ(inf);                                                \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ELEM_TYPE val, uint32_t n) { \
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = SPLAT_MACRO(val);                                                                         \
    ਅਚਲ HVX_ਸਦਿਸ਼ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ਅਣਵਰਤਿਆ(inf);                                                \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ELEM_TYPE val, uint32_t n) { \
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = SPLAT_MACRO(val);                                                                         \
    ਅਚਲ HVX_ਸਦਿਸ਼ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ਅਣਵਰਤਿਆ(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ਆਕਾਰ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS(hvx_ਜੋੜੋ_scalar_f32, HVX_OP_ਜੋੜੋ_SCALAR_F32, hvx_vec_splat_f32, ਦਸ਼ਮਲਵ)
DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, ਦਸ਼ਮਲਵ)
DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, ਦਸ਼ਮਲਵ)

DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS(hvx_ਜੋੜੋ_scalar_f16, HVX_OP_ਜੋੜੋ_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_ਦੁਆਧਾਰੀ_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ELEM_TYPE val, ਅਚਲ uint32_t ਅੰਕ_elems) { \
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128) && hex_is_aligned((ਖਾਲੀ *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, ਅੰਕ_elems);                                                                                   \
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, ਅੰਕ_elems);                                                                                   \
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, ਅੰਕ_elems);                                                                                   \
    } ਨਹੀਂਤਾਂ {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, ਅੰਕ_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_ਦੁਆਧਾਰੀ_SCALAR_DISPATCHER(hvx_ਜੋੜੋ_scalar_f32, ਦਸ਼ਮਲਵ)
HVX_ਦੁਆਧਾਰੀ_SCALAR_DISPATCHER(hvx_sub_scalar_f32, ਦਸ਼ਮਲਵ)
HVX_ਦੁਆਧਾਰੀ_SCALAR_DISPATCHER(hvx_mul_scalar_f32, ਦਸ਼ਮਲਵ)

HVX_ਦੁਆਧਾਰੀ_SCALAR_DISPATCHER(hvx_ਜੋੜੋ_scalar_f16, _Float16)
HVX_ਦੁਆਧਾਰੀ_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_ਦੁਆਧਾਰੀ_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128) && hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        hvx_min_scalar_f32_uu(dst, src, val, ਅੰਕ_elems);
    }
}

// MAX Scalar variants

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_max_scalar_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_max_scalar_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_max_scalar_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_max_scalar_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_max_scalar_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128) && hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        hvx_max_scalar_f32_uu(dst, src, val, ਅੰਕ_elems);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f16(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f16(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f16(val);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ਆਕਾਰ(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_min_scalar_f16(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128) && hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        hvx_min_scalar_f16_uu(dst, src, val, ਅੰਕ_elems);
    }
}

// CLAMP Scalar variants

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_ਸਦਿਸ਼_Pred pred_cap_ਸੱਜਾ = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_ਸਦਿਸ਼_Pred pred_cap_ਖੱਬਾ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_ਸਦਿਸ਼ tmp = Q6_V_vmux_QVV(pred_cap_ਸੱਜਾ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ਖੱਬਾ, min_vec, tmp);                      \
    })

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ min, ਅਚਲ ਦਸ਼ਮਲਵ max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f32(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f32(max);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ min, ਅਚਲ ਦਸ਼ਮਲਵ max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f32(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f32(max);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ min, ਅਚਲ ਦਸ਼ਮਲਵ max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f32(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f32(max);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ min, ਅਚਲ ਦਸ਼ਮਲਵ max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f32(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ min, ਅਚਲ ਦਸ਼ਮਲਵ max, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128) && hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, ਅੰਕ_elems);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_ਸਦਿਸ਼_Pred pred_cap_ਸੱਜਾ = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_ਸਦਿਸ਼_Pred pred_cap_ਖੱਬਾ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_ਸਦਿਸ਼ tmp = Q6_V_vmux_QVV(pred_cap_ਸੱਜਾ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ਖੱਬਾ, min_vec, tmp);                      \
    })

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 min, ਅਚਲ _Float16 max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f16(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f16(max);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 min, ਅਚਲ _Float16 max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f16(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f16(max);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 min, ਅਚਲ _Float16 max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f16(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f16(max);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 min, ਅਚਲ _Float16 max, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ min_vec = hvx_vec_splat_f16(min);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ਆਕਾਰ(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_clamp_scalar_f16(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 min, ਅਚਲ _Float16 max, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128) && hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, ਅੰਕ_elems);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_ਸਦਿਸ਼_Pred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_ਸਦਿਸ਼     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_leaky_relu_scalar_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ ns, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero_vec = hvx_vec_splat_f32(0.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ ns_vec   = hvx_vec_splat_f32(ns);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_leaky_relu_scalar_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ ns, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero_vec = hvx_vec_splat_f32(0.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ ns_vec   = hvx_vec_splat_f32(ns);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_leaky_relu_scalar_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ ns, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero_vec = hvx_vec_splat_f32(0.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ ns_vec   = hvx_vec_splat_f32(ns);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_leaky_relu_scalar_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ ns, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero_vec = hvx_vec_splat_f32(0.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_leaky_relu_scalar_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ ns, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128) && hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, ਅੰਕ_elems);
    } ਨਹੀਂਤਾਂ {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, ਅੰਕ_elems);
    }
}

//
// Abs
//

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_abs_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc = (HVX_ਸਦਿਸ਼ *) src;

    ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t epv       = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t nvec      = n / epv;
    ਅਚਲ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    ਕਰੋ {                                                                   \
        dst_type * ਸੀਮਿਤ vdst  = (dst_type *) dst;                      \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                       \
                                                                           \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(_Float16);                       \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                             \
        ਅਚਲ uint32_t nvec = n / epv;                                     \
        ਅਚਲ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ਲਈ (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        ਜੇ (nloe) {                                                        \
            HVX_ਸਦਿਸ਼ v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);             \
        }                                                                  \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_abs_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_abs_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_abs_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_abs_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_abs_f16(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ uint32_t ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
            hvx_abs_f16_aa(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_abs_f16_au(dst, src, ਅੰਕ_elems);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
            hvx_abs_f16_ua(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_abs_f16_uu(dst, src, ਅੰਕ_elems);
        }
    }
}

//
//ਕਦਮ
//

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_ਕਦਮ_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc = (HVX_ਸਦਿਸ਼ *) src;

    ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t epv       = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t nvec      = n / epv;
    ਅਚਲ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        vdst[i] = hvx_vec_ਕਦਮ_f32(vsrc[i]);
    }
    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v = hvx_vec_ਕਦਮ_f32(vsrc[i]);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_ਕਦਮ_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc = (HVX_ਸਦਿਸ਼ *) src;

    ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(_Float16);
    ਅਚਲ uint32_t epv       = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t nvec      = n / epv;
    ਅਚਲ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        vdst[i] = hvx_vec_ਕਦਮ_f16(vsrc[i]);
    }
    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v = hvx_vec_ਕਦਮ_f16(vsrc[i]);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);
    }
}

//
//ਵਰਗਫਲ
//

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    ਕਰੋ {                                                                   \
        dst_type * ਸੀਮਿਤ vdst  = (dst_type *) dst;                      \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                       \
                                                                           \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);                          \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                             \
        ਅਚਲ uint32_t nvec = n / epv;                                     \
        ਅਚਲ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ਲਈ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ਜੇ (nloe) {                                                        \
            HVX_ਸਦਿਸ਼ v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);             \
        }                                                                  \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ uint32_t ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqr_f32_au(dst, src, ਅੰਕ_elems);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqr_f32_uu(dst, src, ਅੰਕ_elems);
        }
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    ਕਰੋ {                                                                   \
        dst_type * ਸੀਮਿਤ vdst  = (dst_type *) dst;                      \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                       \
                                                                           \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(_Float16);                       \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                             \
        ਅਚਲ uint32_t nvec = n / epv;                                     \
        ਅਚਲ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ਲਈ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ਜੇ (nloe) {                                                        \
            HVX_ਸਦਿਸ਼ v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);             \
        }                                                                  \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqr_f16(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ uint32_t ਅੰਕ_elems) {
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqr_f16_au(dst, src, ਅੰਕ_elems);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqr_f16_uu(dst, src, ਅੰਕ_elems);
        }
    }
}

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_ਜੋੜੋ_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_SUB_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MUL_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_ਜੋੜੋ_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_SUB_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MUL_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ hvx_arith_loop_body
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_ਜੋੜੋ_SCALAR_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_SUB_SCALAR_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MUL_SCALAR_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_ਜੋੜੋ_SCALAR_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_SUB_SCALAR_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MUL_SCALAR_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ hvx_scalar_loop_body
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MIN_SCALAR
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MAX_SCALAR
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MIN_SCALAR_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_CLAMP_SCALAR
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_CLAMP_SCALAR_F16
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_LEAKY_RELU_SCALAR
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ DEFINE_HVX_ਦੁਆਧਾਰੀ_OP_VARIANTS
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ DEFINE_HVX_ਦੁਆਧਾਰੀ_SCALAR_OP_VARIANTS
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_ਦੁਆਧਾਰੀ_DISPATCHER
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ ਅਣਵਰਤਿਆ

#ਜੇ_ਅੰਤ // HVX_ARITH_H
