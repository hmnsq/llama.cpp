#ifndef HVX_SCALE_H
#define HVX_SCALE_H

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব "hvx-base.h"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    করো {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ধ্রুবক uint32_t elem_আকার = আকার(দশমিক);                                    \
        ধ্রুবক uint32_t epv = 128 / elem_আকার;                                        \
        ধ্রুবক uint32_t nvec = n / epv;                                               \
        ধ্রুবক uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        জন্য (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        যদি (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((শূন্য *) &vdst[i], nloe * elem_আকার, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_scale_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale) {
    প্রতিজ্ঞা((size_t) dst % 128 == 0);
    প্রতিজ্ঞা((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale) {
    প্রতিজ্ঞা((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale) {
    প্রতিজ্ঞা((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale) {
    যদি (((size_t) dst & 127) == 0) {
        যদি (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } অন্যথা {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } অন্যথা {
        যদি (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } অন্যথা {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                     \
    করো {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src_type * restrict vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                         \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                        \
                                                                                          \
        ধ্রুবক uint32_t elem_আকার = আকার(দশমিক);                                         \
        ধ্রুবক uint32_t epv = 128 / elem_আকার;                                             \
        ধ্রুবক uint32_t nvec = n / epv;                                                    \
        ধ্রুবক uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        জন্য (; i < nvec; ++i) {                                                           \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        যদি (nloe) {                                                                       \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((শূন্য *) &vdst[i], nloe * elem_আকার, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    প্রতিজ্ঞা((size_t) dst % 128 == 0);
    প্রতিজ্ঞা((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    প্রতিজ্ঞা((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    প্রতিজ্ঞা((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    যদি (((size_t) dst & 127) == 0) {
        যদি (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } অন্যথা {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } অন্যথা {
        যদি (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } অন্যথা {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#define hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)                                                \
    করো {                                                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                                                 \
        src_type * restrict vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                                                   \
                                                                                                                     \
        ধ্রুবক uint32_t nvec = n / VLEN_FP16;                                                                         \
        ধ্রুবক uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        জন্য (; i < nvec; ++i) {                                                                                      \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                                                    \
        }                                                                                                            \
        যদি (nloe) {                                                                                                  \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                                               \
            vec_store((শূন্য *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    প্রতিজ্ঞা((size_t) dst % 128 == 0);
    প্রতিজ্ঞা((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f16_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    প্রতিজ্ঞা((size_t) dst % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f16_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    প্রতিজ্ঞা((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f16_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_scale_offset_f16(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক দশমিক scale, ধ্রুবক দশমিক offset) {
    যদি (((size_t) dst & 127) == 0) {
        যদি (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_aa(dst, src, n, scale, offset);
        } অন্যথা {
            hvx_scale_offset_f16_au(dst, src, n, scale, offset);
        }
    } অন্যথা {
        যদি (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_ua(dst, src, n, scale, offset);
        } অন্যথা {
            hvx_scale_offset_f16_uu(dst, src, n, scale, offset);
        }
    }
}

#endif // HVX_SCALE_H
