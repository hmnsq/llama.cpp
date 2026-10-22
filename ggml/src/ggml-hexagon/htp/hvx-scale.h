#ifndef HVX_SCALE_H
#define HVX_SCALE_H

#ضمن <assert.h>
#ضمن <stddef.h>
#ضمن <stdint.h>

#ضمن "hvx-base.h"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    افعل {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ثابت uint32_t elem_size = حجم(عائم);                                    \
        ثابت uint32_t epv = 128 / elem_size;                                        \
        ثابت uint32_t nvec = n / epv;                                               \
        ثابت uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        لكل (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        إذا (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((فراغ *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } طالما(0)

ساكن مضمن فراغ hvx_scale_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_f32_au(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_f32_ua(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_f32_uu(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_f32(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale) {
    إذا (((size_t) dst & 127) == 0) {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } وإلا {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } وإلا {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } وإلا {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                     \
    افعل {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src_type * restrict vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                         \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                        \
                                                                                          \
        ثابت uint32_t elem_size = حجم(عائم);                                         \
        ثابت uint32_t epv = 128 / elem_size;                                             \
        ثابت uint32_t nvec = n / epv;                                                    \
        ثابت uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        لكل (; i < nvec; ++i) {                                                           \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        إذا (nloe) {                                                                       \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((فراغ *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } طالما(0)

ساكن مضمن فراغ hvx_scale_offset_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_offset_f32_au(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_offset_f32_ua(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_offset_f32_uu(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_offset_f32(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    إذا (((size_t) dst & 127) == 0) {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } وإلا {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } وإلا {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } وإلا {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#define hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)                                                \
    افعل {                                                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                                                 \
        src_type * restrict vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                                                   \
                                                                                                                     \
        ثابت uint32_t nvec = n / VLEN_FP16;                                                                         \
        ثابت uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        لكل (; i < nvec; ++i) {                                                                                      \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                                                    \
        }                                                                                                            \
        إذا (nloe) {                                                                                                  \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                                               \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } طالما(0)

ساكن مضمن فراغ hvx_scale_offset_f16_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_offset_f16_au(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_offset_f16_ua(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_offset_f16_uu(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_offset_f16(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحيح n, ثابت عائم scale, ثابت عائم offset) {
    إذا (((size_t) dst & 127) == 0) {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_aa(dst, src, n, scale, offset);
        } وإلا {
            hvx_scale_offset_f16_au(dst, src, n, scale, offset);
        }
    } وإلا {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_ua(dst, src, n, scale, offset);
        } وإلا {
            hvx_scale_offset_f16_uu(dst, src, n, scale, offset);
        }
    }
}

#endif // HVX_SCALE_H
