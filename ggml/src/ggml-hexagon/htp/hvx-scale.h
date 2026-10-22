#ifndef HVX_SCALE_H
#define HVX_SCALE_H

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

#ଅନ୍ତର୍ଭାଵ "hvx-base.h"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    କର {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ଅଚଳ uint32_t elem_size = ଆକାର(ଦଶମିକ);                                    \
        ଅଚଳ uint32_t epv = 128 / elem_size;                                        \
        ଅଚଳ uint32_t nvec = n / epv;                                               \
        ଅଚଳ uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        ପାଇଁ (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        ଯଦି (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } ଯେତେବେଳେ(0)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale) {
    ପ୍ରତିଜ୍ଞା((size_t) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_f32_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale) {
    ପ୍ରତିଜ୍ଞା((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_f32_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale) {
    ପ୍ରତିଜ୍ଞା((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_f32_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_f32(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale) {
    ଯଦି (((size_t) dst & 127) == 0) {
        ଯଦି (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } ଅନ୍ୟଥା {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } ଅନ୍ୟଥା {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                     \
    କର {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src_type * restrict vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                         \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                        \
                                                                                          \
        ଅଚଳ uint32_t elem_size = ଆକାର(ଦଶମିକ);                                         \
        ଅଚଳ uint32_t epv = 128 / elem_size;                                             \
        ଅଚଳ uint32_t nvec = n / epv;                                                    \
        ଅଚଳ uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ପାଇଁ (; i < nvec; ++i) {                                                           \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        ଯଦି (nloe) {                                                                       \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } ଯେତେବେଳେ(0)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f32_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ପ୍ରତିଜ୍ଞା((size_t) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f32_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ପ୍ରତିଜ୍ଞା((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f32_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ପ୍ରତିଜ୍ଞା((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f32_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f32(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ଯଦି (((size_t) dst & 127) == 0) {
        ଯଦି (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } ଅନ୍ୟଥା {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } ଅନ୍ୟଥା {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#define hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)                                                \
    କର {                                                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                                                 \
        src_type * restrict vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                                                   \
                                                                                                                     \
        ଅଚଳ uint32_t nvec = n / VLEN_FP16;                                                                         \
        ଅଚଳ uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        ପାଇଁ (; i < nvec; ++i) {                                                                                      \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                                                    \
        }                                                                                                            \
        ଯଦି (nloe) {                                                                                                  \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                                               \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } ଯେତେବେଳେ(0)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ପ୍ରତିଜ୍ଞା((size_t) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f16_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ପ୍ରତିଜ୍ଞା((size_t) dst % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f16_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ପ୍ରତିଜ୍ଞା((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f16_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_scale_offset_f16(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ offset) {
    ଯଦି (((size_t) dst & 127) == 0) {
        ଯଦି (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_aa(dst, src, n, scale, offset);
        } ଅନ୍ୟଥା {
            hvx_scale_offset_f16_au(dst, src, n, scale, offset);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_ua(dst, src, n, scale, offset);
        } ଅନ୍ୟଥା {
            hvx_scale_offset_f16_uu(dst, src, n, scale, offset);
        }
    }
}

#endif // HVX_SCALE_H
