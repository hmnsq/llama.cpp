#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_ARITH_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_ARITH_H

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <math.h>

#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಳಸದ(x) (ಶೂನ್ಯ)(x)

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_ಗಾತ್ರ, vec_store, vec_op) \
    ಮಾಡು {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / (elem_ಗಾತ್ರ);                                          \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                                    \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ಗಾಗಿ (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        ಆದರೆ (nloe) {                                                                       \
            HVX_ಸದಿಶ v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * (elem_ಗಾತ್ರ), v);                          \
        }                                                                                 \
    } ಆಗ(0)

#ಆದರೆ __HVX_ARCH__ < 79

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#ಇಲ್ಲದಿದ್ದರೆ

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_F16(a, b) hvx_vec_ಕೂಡಿಸು_f16_f16(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_aaa(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_aau(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_aua(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_ಸದಿಶ, HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_auu(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_ಸದಿಶ, HVX_UVector, HVX_UVector, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uaa(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uau(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uua(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uuu(uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_ಕೂಡಿಸು_f32, HVX_OP_ಕೂಡಿಸು_F32, ದಶಮಾಂಶ)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, ದಶಮಾಂಶ)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, ದಶಮಾಂಶ)

DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_ಕೂಡಿಸು_f16, HVX_OP_ಕೂಡಿಸು_F16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_ದ್ವಿಮಾನ_DISPATCHER(OP_NAME)                                                                                                       \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) { \
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {                                                                                                 \
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src0, 128)) {                                                                                            \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_aau(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                                             \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_auu(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        }                                                                                                                                    \
    } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                                                 \
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src0, 128)) {                                                                                            \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_uau(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                                             \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_uuu(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_ಕೂಡಿಸು_f32)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_sub_f32)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_mul_f32)

HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_ಕೂಡಿಸು_f16)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_sub_f16)
HVX_ದ್ವಿಮಾನ_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_mul_mul_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src2, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src1 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src2 % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst  = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc0 = (HVX_ಸದಿಶ *) src0;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc1 = (HVX_ಸದಿಶ *) src1;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc2 = (HVX_ಸದಿಶ *) src2;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nvec = ಅಂಕೆ_elems / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = ಅಂಕೆ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_ಸದಿಶ v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v2);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_mul_mul_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src2, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src1 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src2 % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst  = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc0 = (HVX_ಸದಿಶ *) src0;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc1 = (HVX_ಸದಿಶ *) src1;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc2 = (HVX_ಸದಿಶ *) src2;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nvec = ಅಂಕೆ_elems / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = ಅಂಕೆ_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_ಸದಿಶ v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v2);
    }
}

// Scalar Operations

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_scalar_loop_body(dst_type, src_type, elem_ಗಾತ್ರ, vec_store, scalar_op_macro) \
    ಮಾಡು {                                                                                \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                    \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                    \
                                                                                        \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / (elem_ಗಾತ್ರ);                                        \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                                  \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        ಗಾಗಿ (; i < nvec; i++) {                                                         \
            HVX_ಸದಿಶ v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        ಆದರೆ (nloe) {                                                                     \
            HVX_ಸದಿಶ v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * (elem_ಗಾತ್ರ), v);                        \
        }                                                                               \
    } ಆಗ(0)

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_SCALAR_F32(v)                                   \
    ({                                                             \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_ಸದಿಶ ಹೊರಹೋಗು = HVX_OP_ಕೂಡಿಸು_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, ಹೊರಹೋಗು);                         \
    })

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_SCALAR_F16(v)                                   \
    ({                                                             \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_ಸದಿಶ ಹೊರಹೋಗು = HVX_OP_ಕೂಡಿಸು_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, ಹೊರಹೋಗು);                         \
    })

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ELEM_TYPE val, uint32_t n) { \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = SPLAT_MACRO(val);                                                                         \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ಬಳಸದ(inf);                                                \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                  \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ELEM_TYPE val, uint32_t n) { \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = SPLAT_MACRO(val);                                                                         \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ಬಳಸದ(inf);                                                \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ELEM_TYPE val, uint32_t n) { \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = SPLAT_MACRO(val);                                                                         \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ಬಳಸದ(inf);                                                \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ELEM_TYPE val, uint32_t n) { \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = SPLAT_MACRO(val);                                                                         \
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); ಬಳಸದ(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಗಾತ್ರ(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_ಕೂಡಿಸು_scalar_f32, HVX_OP_ಕೂಡಿಸು_SCALAR_F32, hvx_vec_splat_f32, ದಶಮಾಂಶ)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, ದಶಮಾಂಶ)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, ದಶಮಾಂಶ)

DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_ಕೂಡಿಸು_scalar_f16, HVX_OP_ಕೂಡಿಸು_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ELEM_TYPE val, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) { \
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, ಅಂಕೆ_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_ಕೂಡಿಸು_scalar_f32, ದಶಮಾಂಶ)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_sub_scalar_f32, ದಶಮಾಂಶ)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_mul_scalar_f32, ದಶಮಾಂಶ)

HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_ಕೂಡಿಸು_scalar_f16, _Float16)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_ದ್ವಿಮಾನ_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_min_scalar_f32_uu(dst, src, val, ಅಂಕೆ_elems);
    }
}

// MAX Scalar variants

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_max_scalar_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_max_scalar_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_max_scalar_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_max_scalar_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_max_scalar_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_max_scalar_f32_uu(dst, src, val, ಅಂಕೆ_elems);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f16(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f16(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f16(val);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಗಾತ್ರ(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_min_scalar_f16(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_min_scalar_f16_uu(dst, src, val, ಅಂಕೆ_elems);
    }
}

// CLAMP Scalar variants

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_ಸದಿಶ_Pred pred_cap_ಬಲ = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_ಸದಿಶ_Pred pred_cap_ಎಡ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_ಸದಿಶ tmp = Q6_V_vmux_QVV(pred_cap_ಬಲ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ಎಡ, min_vec, tmp);                      \
    })

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f32(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f32(max);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f32(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f32(max);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f32(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f32(max);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f32(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, ಅಂಕೆ_elems);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_ಸದಿಶ_Pred pred_cap_ಬಲ = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_ಸದಿಶ_Pred pred_cap_ಎಡ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_ಸದಿಶ tmp = Q6_V_vmux_QVV(pred_cap_ಬಲ, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ಎಡ, min_vec, tmp);                      \
    })

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 min, ಸ್ಥಿರಾಂಕ _Float16 max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f16(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f16(max);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 min, ಸ್ಥಿರಾಂಕ _Float16 max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f16(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f16(max);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 min, ಸ್ಥಿರಾಂಕ _Float16 max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f16(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f16(max);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 min, ಸ್ಥಿರಾಂಕ _Float16 max, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_vec = hvx_vec_splat_f16(min);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಗಾತ್ರ(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_clamp_scalar_f16(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 min, ಸ್ಥಿರಾಂಕ _Float16 max, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, ಅಂಕೆ_elems);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_ಸದಿಶ_Pred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_ಸದಿಶ     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ns, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero_vec = hvx_vec_splat_f32(0.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ns_vec   = hvx_vec_splat_f32(ns);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ns, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero_vec = hvx_vec_splat_f32(0.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ns_vec   = hvx_vec_splat_f32(ns);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_ಸದಿಶ, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ns, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero_vec = hvx_vec_splat_f32(0.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ns_vec   = hvx_vec_splat_f32(ns);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_ಸದಿಶ, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ns, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero_vec = hvx_vec_splat_f32(0.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_leaky_relu_scalar_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ns, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128) && hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, ಅಂಕೆ_elems);
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, ಅಂಕೆ_elems);
    }
}

//
// Abs
//

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_abs_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t epv       = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nvec      = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    ಮಾಡು {                                                                   \
        dst_type * ಸೀಮಿತ vdst  = (dst_type *) dst;                      \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                       \
                                                                           \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(_Float16);                       \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                             \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                     \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ಗಾಗಿ (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        ಆದರೆ (nloe) {                                                        \
            HVX_ಸದಿಶ v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);             \
        }                                                                  \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_abs_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_abs_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_abs_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_abs_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_abs_f16(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_abs_f16_aa(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_abs_f16_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_abs_f16_ua(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_abs_f16_uu(dst, src, ಅಂಕೆ_elems);
        }
    }
}

//
//ಹೆಜ್ಜೆ
//

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_ಹೆಜ್ಜೆ_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t epv       = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nvec      = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_ಹೆಜ್ಜೆ_f32(vsrc[i]);
    }
    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ v = hvx_vec_ಹೆಜ್ಜೆ_f32(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_ಹೆಜ್ಜೆ_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ uint32_t epv       = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nvec      = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_ಹೆಜ್ಜೆ_f16(vsrc[i]);
    }
    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ v = hvx_vec_ಹೆಜ್ಜೆ_f16(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);
    }
}

//
//ವರ್ಗಫಲ
//

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    ಮಾಡು {                                                                   \
        dst_type * ಸೀಮಿತ vdst  = (dst_type *) dst;                      \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                       \
                                                                           \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);                          \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                             \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                     \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ಗಾಗಿ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ಆದರೆ (nloe) {                                                        \
            HVX_ಸದಿಶ v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);             \
        }                                                                  \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqr_f32_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqr_f32_uu(dst, src, ಅಂಕೆ_elems);
        }
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    ಮಾಡು {                                                                   \
        dst_type * ಸೀಮಿತ vdst  = (dst_type *) dst;                      \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                       \
                                                                           \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(_Float16);                       \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                             \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                     \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        ಗಾಗಿ (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ಆದರೆ (nloe) {                                                        \
            HVX_ಸದಿಶ v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);             \
        }                                                                  \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqr_f16(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) {
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqr_f16_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqr_f16_uu(dst, src, ಅಂಕೆ_elems);
        }
    }
}

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_ಕೂಡಿಸು_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_SUB_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MUL_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_ಕೂಡಿಸು_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_SUB_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MUL_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು hvx_arith_loop_body
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_ಕೂಡಿಸು_SCALAR_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_SUB_SCALAR_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MUL_SCALAR_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_ಕೂಡಿಸು_SCALAR_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_SUB_SCALAR_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MUL_SCALAR_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು hvx_scalar_loop_body
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MIN_SCALAR
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MAX_SCALAR
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MIN_SCALAR_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_CLAMP_SCALAR
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_CLAMP_SCALAR_F16
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_LEAKY_RELU_SCALAR
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು DEFINE_HVX_ದ್ವಿಮಾನ_OP_VARIANTS
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು DEFINE_HVX_ದ್ವಿಮಾನ_SCALAR_OP_VARIANTS
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_ದ್ವಿಮಾನ_DISPATCHER
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು ಬಳಸದ

#ಆದರೆ_ಅಂತ್ಯ // HVX_ARITH_H
