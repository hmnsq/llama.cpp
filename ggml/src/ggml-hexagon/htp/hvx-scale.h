#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_SCALE_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_SCALE_H

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hvx-base.h"

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    ಮಾಡು {                                                                             \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                 \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_ಸದಿಶ vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);                                    \
        ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / elem_ಗಾತ್ರ;                                        \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                               \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        ಗಾಗಿ (; i < nvec; ++i) {                                                      \
            HVX_ಸದಿಶ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        ಆದರೆ (nloe) {                                                                  \
            HVX_ಸದಿಶ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale) {
    ಪ್ರತಿಜ್ಞಾ((size_t) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale) {
    ಪ್ರತಿಜ್ಞಾ((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale) {
    ಪ್ರತಿಜ್ಞಾ((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale) {
    ಆದರೆ (((size_t) dst & 127) == 0) {
        ಆದರೆ (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_scale_ಸ್ಥಾನಾಂತರ_f32_loop_body(dst_type, src_type, vec_store)                     \
    ಮಾಡು {                                                                                  \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                      \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_ಸದಿಶ vs = hvx_vec_splat_f32(scale);                                         \
        HVX_ಸದಿಶ vo = hvx_vec_splat_f32(ಸ್ಥಾನಾಂತರ);                                        \
                                                                                          \
        ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);                                         \
        ಸ್ಥಿರಾಂಕ uint32_t epv = 128 / elem_ಗಾತ್ರ;                                             \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;                                                    \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ಗಾಗಿ (; i < nvec; ++i) {                                                           \
            HVX_ಸದಿಶ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        ಆದರೆ (nloe) {                                                                       \
            HVX_ಸದಿಶ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಪ್ರತಿಜ್ಞಾ((size_t) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((size_t) src % 128 == 0);
    hvx_scale_ಸ್ಥಾನಾಂತರ_f32_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f32_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಪ್ರತಿಜ್ಞಾ((size_t) dst % 128 == 0);
    hvx_scale_ಸ್ಥಾನಾಂತರ_f32_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f32_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಪ್ರತಿಜ್ಞಾ((size_t) src % 128 == 0);
    hvx_scale_ಸ್ಥಾನಾಂತರ_f32_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f32_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    hvx_scale_ಸ್ಥಾನಾಂತರ_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಆದರೆ (((size_t) dst & 127) == 0) {
        ಆದರೆ (((size_t) src & 127) == 0) {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f32_aa(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f32_au(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (((size_t) src & 127) == 0) {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f32_ua(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f32_uu(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#ವ್ಯಾಖ್ಯಾನಿಸು hvx_scale_ಸ್ಥಾನಾಂತರ_f16_loop_body(dst_type, src_type, vec_store)                                                \
    ಮಾಡು {                                                                                                             \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                                                 \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_ಸದಿಶ vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_ಸದಿಶ vo = hvx_vec_splat_f32(ಸ್ಥಾನಾಂತರ);                                                                   \
                                                                                                                     \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP16;                                                                         \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        ಗಾಗಿ (; i < nvec; ++i) {                                                                                      \
            HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(vsrc[i]);                                                          \
            HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);                                                                    \
        }                                                                                                            \
        ಆದರೆ (nloe) {                                                                                                  \
            HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(vsrc[i]);                                                          \
            HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_ಸದಿಶ v = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);                                                               \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಪ್ರತಿಜ್ಞಾ((size_t) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((size_t) src % 128 == 0);
    hvx_scale_ಸ್ಥಾನಾಂತರ_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಪ್ರತಿಜ್ಞಾ((size_t) dst % 128 == 0);
    hvx_scale_ಸ್ಥಾನಾಂತರ_f16_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಪ್ರತಿಜ್ಞಾ((size_t) src % 128 == 0);
    hvx_scale_ಸ್ಥಾನಾಂತರ_f16_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    hvx_scale_ಸ್ಥಾನಾಂತರ_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_ಸ್ಥಾನಾಂತರ_f16(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸ್ಥಾನಾಂತರ) {
    ಆದರೆ (((size_t) dst & 127) == 0) {
        ಆದರೆ (((size_t) src & 127) == 0) {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f16_aa(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f16_au(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (((size_t) src & 127) == 0) {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f16_ua(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_scale_ಸ್ಥಾನಾಂತರ_f16_uu(dst, src, n, scale, ಸ್ಥಾನಾಂತರ);
        }
    }
}

#ಆದರೆ_ಅಂತ್ಯ // HVX_SCALE_H
