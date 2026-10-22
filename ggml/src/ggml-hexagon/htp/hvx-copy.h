#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_COPY_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_COPY_H

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hvx-base.h"

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_splat_pragma(x) _Pragma(#x)
#ವ್ಯಾಖ್ಯಾನಿಸು hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    ಮಾಡು {                                                         \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_ಗಾತ್ರ);                   \
        uint32_t nloe = n % (128 / elem_ಗಾತ್ರ);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        ಗಾಗಿ (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        ಆದರೆ (nloe) {                                              \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, src); \
        }                                                        \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_a(ಶೂನ್ಯ * ಸೀಮಿತ dst, HVX_ಸದಿಶ src, uint32_t n, uint32_t elem_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_splat_loop_body(HVX_ಸದಿಶ, hvx_vec_store_a, 4);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_u(ಶೂನ್ಯ * ಸೀಮಿತ dst, HVX_ಸದಿಶ src, uint32_t n, uint32_t elem_ಗಾತ್ರ) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_f32_a(ಶೂನ್ಯ * ಸೀಮಿತ dst, ದಶಮಾಂಶ v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, ಗಾತ್ರ(ದಶಮಾಂಶ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_f32_u(ಶೂನ್ಯ * ಸೀಮಿತ dst, ದಶಮಾಂಶ v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, ಗಾತ್ರ(ದಶಮಾಂಶ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_f16_a(ಶೂನ್ಯ * ಸೀಮಿತ dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, ಗಾತ್ರ(__fp16));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_f16_u(ಶೂನ್ಯ * ಸೀಮಿತ dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, ಗಾತ್ರ(__fp16));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_u16_a(ಶೂನ್ಯ * ಸೀಮಿತ dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, ಗಾತ್ರ(uint16_t));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_u16_u(ಶೂನ್ಯ * ಸೀಮಿತ dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, ಗಾತ್ರ(uint16_t));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_u8_a(ಶೂನ್ಯ * ಸೀಮಿತ dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_splat_u8_u(ಶೂನ್ಯ * ಸೀಮಿತ dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    ಮಾಡು {                                                             \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                 \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                 \
                                                                     \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                       \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                               \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        ಗಾಗಿ (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        ಆದರೆ (nloe) {                                                  \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, vsrc[i]); \
        }                                                            \
    } ಆಗ(0)

// Generic copy routines
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n, uint32_t elem_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n, uint32_t elem_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_copy_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n, uint32_t elem_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n, uint32_t elem_ಗಾತ್ರ) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_aa(dst, src, n, ಗಾತ್ರ(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_au(dst, src, n, ಗಾತ್ರ(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_ua(dst, src, n, ಗಾತ್ರ(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_uu(dst, src, n, ಗಾತ್ರ(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_aa(dst, src, n, ಗಾತ್ರ(ದಶಮಾಂಶ));
}

// copy n fp32 elements : destination is aligned, source is unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_au(dst, src, n, ಗಾತ್ರ(ದಶಮಾಂಶ));
}

// copy n fp32 elements : destination is unaligned, source is aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_ua(dst, src, n, ಗಾತ್ರ(ದಶಮಾಂಶ));
}

// copy n fp32 elements : destination and source are unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_uu(dst, src, n, ಗಾತ್ರ(ದಶಮಾಂಶ));
}

//// fp32 -> fp16

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    ಮಾಡು {                                                                            \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                \
                                                                                    \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(__fp16);                                  \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                                      \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                              \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        ಗಾಗಿ (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_ಗಮ್ಯ_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        ಆದರೆ (nloe) {                                                                 \
            HVX_ಸದಿಶ v = hvx_vec_f32_ಗಮ್ಯ_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);                      \
        }                                                                           \
    } ಆಗ(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f16_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    ಮಾಡು {                                                                            \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                \
                                                                                    \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(__fp16);                                  \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                                      \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        ಗಾಗಿ (i = 0; i < nvec; ++i) {                                                \
            HVX_ಸದಿಶ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        ಆದರೆ (nloe) {                                                                 \
            HVX_ಸದಿಶ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_ಸದಿಶ vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            ಆದರೆ (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            ಆದರೆ (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), vd);                \
            }                                                                       \
        }                                                                           \
    } ಆಗ(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    ಮಾಡು {                                                          \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;              \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;              \
                                                                  \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(int32_t);               \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                    \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                            \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        ಗಾಗಿ (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        ಆದರೆ (nloe) {                                               \
            HVX_ಸದಿಶ v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);    \
        }                                                         \
    } ಆಗ(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_i32_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_i32_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    ಮಾಡು {                                                          \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;              \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;              \
                                                                  \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);                 \
        ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / elem_ಗಾತ್ರ;                    \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                            \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        ಗಾಗಿ (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        ಆದರೆ (nloe) {                                               \
            HVX_ಸದಿಶ v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);    \
        }                                                         \
    } ಆಗ(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_i32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_copy_f32_i32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#ಆದರೆ_ಅಂತ್ಯ // HVX_COPY_H
