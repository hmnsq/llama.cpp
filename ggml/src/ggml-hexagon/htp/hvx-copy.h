#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_COPY_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_COPY_H

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_splat_pragma(x) _Pragma(#x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    ਕਰੋ {                                                         \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_ਮਾਪ);                   \
        uint32_t nloe = n % (128 / elem_ਮਾਪ);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        ਲਈ (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        ਜੇ (nloe) {                                              \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, src); \
        }                                                        \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_a(ਖਾਲੀ * ਸੀਮਿਤ dst, HVX_ਸਦਿਸ਼ src, uint32_t n, uint32_t elem_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_splat_loop_body(HVX_ਸਦਿਸ਼, hvx_vec_store_a, 4);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_u(ਖਾਲੀ * ਸੀਮਿਤ dst, HVX_ਸਦਿਸ਼ src, uint32_t n, uint32_t elem_ਮਾਪ) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_f32_a(ਖਾਲੀ * ਸੀਮਿਤ dst, ਦਸ਼ਮਲਵ v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_f32_u(ਖਾਲੀ * ਸੀਮਿਤ dst, ਦਸ਼ਮਲਵ v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_f16_a(ਖਾਲੀ * ਸੀਮਿਤ dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, ਆਕਾਰ(__fp16));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_f16_u(ਖਾਲੀ * ਸੀਮਿਤ dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, ਆਕਾਰ(__fp16));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_u16_a(ਖਾਲੀ * ਸੀਮਿਤ dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, ਆਕਾਰ(uint16_t));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_u16_u(ਖਾਲੀ * ਸੀਮਿਤ dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, ਆਕਾਰ(uint16_t));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_u8_a(ਖਾਲੀ * ਸੀਮਿਤ dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_splat_u8_u(ਖਾਲੀ * ਸੀਮਿਤ dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    ਕਰੋ {                                                             \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                 \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                 \
                                                                     \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                       \
        ਅਚਲ uint32_t nvec = n / epv;                               \
        ਅਚਲ uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        ਲਈ (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        ਜੇ (nloe) {                                                  \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, vsrc[i]); \
        }                                                            \
    } ਜਦੋਂ(0)

// Generic copy routines
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n, uint32_t elem_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n, uint32_t elem_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_copy_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n, uint32_t elem_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n, uint32_t elem_ਮਾਪ) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_aa(dst, src, n, ਆਕਾਰ(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_au(dst, src, n, ਆਕਾਰ(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_ua(dst, src, n, ਆਕਾਰ(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_uu(dst, src, n, ਆਕਾਰ(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_aa(dst, src, n, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

// copy n fp32 elements : destination is aligned, source is unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_au(dst, src, n, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

// copy n fp32 elements : destination is unaligned, source is aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_ua(dst, src, n, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

// copy n fp32 elements : destination and source are unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_uu(dst, src, n, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
}

//// fp32 -> fp16

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    ਕਰੋ {                                                                            \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                \
                                                                                    \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(__fp16);                                  \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                                      \
        ਅਚਲ uint32_t nvec = n / epv;                                              \
        ਅਚਲ uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        ਲਈ (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        ਜੇ (nloe) {                                                                 \
            HVX_ਸਦਿਸ਼ v = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);                      \
        }                                                                           \
    } ਜਦੋਂ(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_f32_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_f32_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f16_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    ਕਰੋ {                                                                            \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                \
                                                                                    \
        ਅਚਲ HVX_ਸਦਿਸ਼ one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(__fp16);                                  \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                                      \
        ਅਚਲ uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        ਲਈ (i = 0; i < nvec; ++i) {                                                \
            HVX_ਸਦਿਸ਼_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        ਜੇ (nloe) {                                                                 \
            HVX_ਸਦਿਸ਼_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_ਸਦਿਸ਼ vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            ਜੇ (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            ਜੇ (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vd);                \
            }                                                                       \
        }                                                                           \
    } ਜਦੋਂ(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    ਕਰੋ {                                                          \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;              \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;              \
                                                                  \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(int32_t);               \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                    \
        ਅਚਲ uint32_t nvec = n / epv;                            \
        ਅਚਲ uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        ਲਈ (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        ਜੇ (nloe) {                                               \
            HVX_ਸਦਿਸ਼ v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);    \
        }                                                         \
    } ਜਦੋਂ(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_i32_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_i32_f32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    ਕਰੋ {                                                          \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;              \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;              \
                                                                  \
        ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);                 \
        ਅਚਲ uint32_t epv  = 128 / elem_ਮਾਪ;                    \
        ਅਚਲ uint32_t nvec = n / epv;                            \
        ਅਚਲ uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        ਲਈ (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        ਜੇ (nloe) {                                               \
            HVX_ਸਦਿਸ਼ v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);    \
        }                                                         \
    } ਜਦੋਂ(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_i32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_copy_f32_i32_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#ਜੇ_ਅੰਤ // HVX_COPY_H
