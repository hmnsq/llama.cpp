#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_SCALE_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_SCALE_H

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    ਕਰੋ {                                                                             \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                 \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_ਸਦਿਸ਼ vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);                                    \
        ਅਚਲ uint32_t epv = 128 / elem_ਮਾਪ;                                        \
        ਅਚਲ uint32_t nvec = n / epv;                                               \
        ਅਚਲ uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        ਲਈ (; i < nvec; ++i) {                                                      \
            HVX_ਸਦਿਸ਼ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        ਜੇ (nloe) {                                                                  \
            HVX_ਸਦਿਸ਼ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale) {
    ਜੇ (((size_t) dst & 127) == 0) {
        ਜੇ (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } ਨਹੀਂਤਾਂ {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } ਨਹੀਂਤਾਂ {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_scale_ਵਿੱਥ_f32_loop_body(dst_type, src_type, vec_store)                     \
    ਕਰੋ {                                                                                  \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                      \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_ਸਦਿਸ਼ vs = hvx_vec_splat_f32(scale);                                         \
        HVX_ਸਦਿਸ਼ vo = hvx_vec_splat_f32(ਵਿੱਥ);                                        \
                                                                                          \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);                                         \
        ਅਚਲ uint32_t epv = 128 / elem_ਮਾਪ;                                             \
        ਅਚਲ uint32_t nvec = n / epv;                                                    \
        ਅਚਲ uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ਲਈ (; i < nvec; ++i) {                                                           \
            HVX_ਸਦਿਸ਼ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        ਜੇ (nloe) {                                                                       \
            HVX_ਸਦਿਸ਼ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) src % 128 == 0);
    hvx_scale_ਵਿੱਥ_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) dst % 128 == 0);
    hvx_scale_ਵਿੱਥ_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) src % 128 == 0);
    hvx_scale_ਵਿੱਥ_f32_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    hvx_scale_ਵਿੱਥ_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਜੇ (((size_t) dst & 127) == 0) {
        ਜੇ (((size_t) src & 127) == 0) {
            hvx_scale_ਵਿੱਥ_f32_aa(dst, src, n, scale, ਵਿੱਥ);
        } ਨਹੀਂਤਾਂ {
            hvx_scale_ਵਿੱਥ_f32_au(dst, src, n, scale, ਵਿੱਥ);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (((size_t) src & 127) == 0) {
            hvx_scale_ਵਿੱਥ_f32_ua(dst, src, n, scale, ਵਿੱਥ);
        } ਨਹੀਂਤਾਂ {
            hvx_scale_ਵਿੱਥ_f32_uu(dst, src, n, scale, ਵਿੱਥ);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_scale_ਵਿੱਥ_f16_loop_body(dst_type, src_type, vec_store)                                                \
    ਕਰੋ {                                                                                                             \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                                                 \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_ਸਦਿਸ਼ vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_ਸਦਿਸ਼ vo = hvx_vec_splat_f32(ਵਿੱਥ);                                                                   \
                                                                                                                     \
        ਅਚਲ uint32_t nvec = n / VLEN_FP16;                                                                         \
        ਅਚਲ uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        ਲਈ (; i < nvec; ++i) {                                                                                      \
            HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vsrc[i]);                                                          \
            HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);                                                                    \
        }                                                                                                            \
        ਜੇ (nloe) {                                                                                                  \
            HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(vsrc[i]);                                                          \
            HVX_ਸਦਿਸ਼ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_ਸਦਿਸ਼ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_ਸਦਿਸ਼ v = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(r0, r1);                                                               \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) src % 128 == 0);
    hvx_scale_ਵਿੱਥ_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) dst % 128 == 0);
    hvx_scale_ਵਿੱਥ_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) src % 128 == 0);
    hvx_scale_ਵਿੱਥ_f16_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    hvx_scale_ਵਿੱਥ_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_ਵਿੱਥ_f16(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ ਵਿੱਥ) {
    ਜੇ (((size_t) dst & 127) == 0) {
        ਜੇ (((size_t) src & 127) == 0) {
            hvx_scale_ਵਿੱਥ_f16_aa(dst, src, n, scale, ਵਿੱਥ);
        } ਨਹੀਂਤਾਂ {
            hvx_scale_ਵਿੱਥ_f16_au(dst, src, n, scale, ਵਿੱਥ);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (((size_t) src & 127) == 0) {
            hvx_scale_ਵਿੱਥ_f16_ua(dst, src, n, scale, ਵਿੱਥ);
        } ਨਹੀਂਤਾਂ {
            hvx_scale_ਵਿੱਥ_f16_uu(dst, src, n, scale, ਵਿੱਥ);
        }
    }
}

#ਜੇ_ਅੰਤ // HVX_SCALE_H
