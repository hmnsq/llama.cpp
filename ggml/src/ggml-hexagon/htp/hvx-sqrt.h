#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_SQRT_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_SQRT_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"

#ಅನ್ತರ್ಭಾವ "hvx-base.h"

#ವ್ಯಾಖ್ಯಾನಿಸು RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#ವ್ಯಾಖ್ಯಾನಿಸು RSQRT_ONE_HALF     0x3f000000  // 0.5
#ವ್ಯಾಖ್ಯಾನಿಸು RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#ಆದರೆ __HVX_ARCH__ < 79
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_rsqrt_f32(HVX_ಸದಿಶ ಒಳ_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_ಸದಿಶ rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_ಸದಿಶ onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_ಸದಿಶ threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_ಸದಿಶ x2, y, ypower2, ತಾತ್ಕಾಲಿಕ;

    x2 = Q6_Vqf32_vmpy_VsfVsf(ಒಳ_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(ಒಳ_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(ತಾತ್ಕಾಲಿಕ));
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(ತಾತ್ಕಾಲಿಕ));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(ತಾತ್ಕಾಲಿಕ, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(ತಾತ್ಕಾಲಿಕ));
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, ತಾತ್ಕಾಲಿಕ);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(ತಾತ್ಕಾಲಿಕ, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(ತಾತ್ಕಾಲಿಕ));
    ತಾತ್ಕಾಲಿಕ    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, ತಾತ್ಕಾಲಿಕ);

    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(ತಾತ್ಕಾಲಿಕ);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#ವ್ಯಾಖ್ಯಾನಿಸು hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    ಮಾಡು {                                                                     \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                         \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                         \
                                                                             \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP32;                                 \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        ಗಾಗಿ (; i < nvec; i++) {                                              \
            HVX_ಸದಿಶ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_ಸದಿಶ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        ಆದರೆ (nloe) {                                                          \
            HVX_ಸದಿಶ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_ಸದಿಶ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0) {
        ಆದರೆ ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqrt_f32_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqrt_f32_uu(dst, src, ಅಂಕೆ_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#ವ್ಯಾಖ್ಯಾನಿಸು hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    ಮಾಡು {                                                                               \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                   \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                   \
                                                                                       \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP16;                                           \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        ಗಾಗಿ (; i < nvec; i++) {                                                        \
            HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(vsrc[i]);                            \
            HVX_ಸದಿಶ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_ಸದಿಶ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);                                      \
        }                                                                              \
        ಆದರೆ (nloe) {                                                                    \
            HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(vsrc[i]);                            \
            HVX_ಸದಿಶ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_ಸದಿಶ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_ಸದಿಶ v = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);                                 \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_sqrt_f16(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಆದರೆ ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0) {
        ಆದರೆ ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqrt_f16_au(dst, src, ಅಂಕೆ_elems);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, ಅಂಕೆ_elems);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_sqrt_f16_uu(dst, src, ಅಂಕೆ_elems);
        }
    }
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_SQRT_H */
