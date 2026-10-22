#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_SQRT_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_SQRT_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RSQRT_ONE_HALF     0x3f000000  // 0.5
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#ਜੇ __HVX_ARCH__ < 79
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_rsqrt_f32(HVX_ਸਦਿਸ਼ ਵਿੱਚ_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_ਸਦਿਸ਼ rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_ਸਦਿਸ਼ onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_ਸਦਿਸ਼ threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_ਸਦਿਸ਼ x2, y, ypower2, ਆਰਜ਼ੀ;

    x2 = Q6_Vqf32_vmpy_VsfVsf(ਵਿੱਚ_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(ਵਿੱਚ_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    ਆਰਜ਼ੀ    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    ਆਰਜ਼ੀ    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(ਆਰਜ਼ੀ));
    ਆਰਜ਼ੀ    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(ਆਰਜ਼ੀ));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(ਆਰਜ਼ੀ, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    ਆਰਜ਼ੀ    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    ਆਰਜ਼ੀ    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(ਆਰਜ਼ੀ));
    ਆਰਜ਼ੀ    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, ਆਰਜ਼ੀ);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(ਆਰਜ਼ੀ, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    ਆਰਜ਼ੀ    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    ਆਰਜ਼ੀ    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(ਆਰਜ਼ੀ));
    ਆਰਜ਼ੀ    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, ਆਰਜ਼ੀ);

    ਵਾਪਸ Q6_Vsf_equals_Vqf32(ਆਰਜ਼ੀ);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    ਕਰੋ {                                                                     \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                         \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                         \
                                                                             \
        ਅਚਲ uint32_t nvec = n / VLEN_FP32;                                 \
        ਅਚਲ uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        ਲਈ (; i < nvec; i++) {                                              \
            HVX_ਸਦਿਸ਼ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_ਸਦਿਸ਼ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        ਜੇ (nloe) {                                                          \
            HVX_ਸਦਿਸ਼ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_ਸਦਿਸ਼ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0) {
        ਜੇ ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqrt_f32_au(dst, src, ਅੰਕ_elems);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqrt_f32_uu(dst, src, ਅੰਕ_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    ਕਰੋ {                                                                               \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                   \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                   \
                                                                                       \
        ਅਚਲ uint32_t nvec = n / VLEN_FP16;                                           \
        ਅਚਲ uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        ਲਈ (; i < nvec; i++) {                                                        \
            HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vsrc[i]);                            \
            HVX_ਸਦਿਸ਼ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_ਸਦਿਸ਼ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);                                      \
        }                                                                              \
        ਜੇ (nloe) {                                                                    \
            HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vsrc[i]);                            \
            HVX_ਸਦਿਸ਼ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_ਸਦਿਸ਼ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_ਸਦਿਸ਼ v = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);                                 \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_sqrt_f16(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਜੇ ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0) {
        ਜੇ ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqrt_f16_au(dst, src, ਅੰਕ_elems);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, ਅੰਕ_elems);
        } ਨਹੀਂਤਾਂ {
            hvx_sqrt_f16_uu(dst, src, ਅੰਕ_elems);
        }
    }
}

#ਜੇ_ਅੰਤ /* HVX_SQRT_H */
