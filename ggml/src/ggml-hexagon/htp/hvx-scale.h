#جے_بنا_تعریف HVX_SCALE_H
#تعریف_کرو HVX_SCALE_H

#ضمن <assert.h>
#ضمن <stddef.h>
#ضمن <stdint.h>

#ضمن "hvx-base.h"

#تعریف_کرو hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    افعل {                                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                                 \
        src_type * محدود vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_سمتیہ vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ثابت uint32_t elem_ناپ = حجم(عائم);                                    \
        ثابت uint32_t epv = 128 / elem_ناپ;                                        \
        ثابت uint32_t nvec = n / epv;                                               \
        ثابت uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        لكل (; i < nvec; ++i) {                                                      \
            HVX_سمتیہ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        إذا (nloe) {                                                                  \
            HVX_سمتیہ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((فراغ *) &vdst[i], nloe * elem_ناپ, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } طالما(0)

ساكن مضمن فراغ hvx_scale_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale) {
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

#تعریف_کرو hvx_scale_وتھ_f32_loop_body(dst_type, src_type, vec_store)                     \
    افعل {                                                                                  \
        dst_type * محدود vdst = (dst_type *) dst;                                      \
        src_type * محدود vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_سمتیہ vs = hvx_vec_splat_f32(scale);                                         \
        HVX_سمتیہ vo = hvx_vec_splat_f32(وتھ);                                        \
                                                                                          \
        ثابت uint32_t elem_ناپ = حجم(عائم);                                         \
        ثابت uint32_t epv = 128 / elem_ناپ;                                             \
        ثابت uint32_t nvec = n / epv;                                                    \
        ثابت uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        لكل (; i < nvec; ++i) {                                                           \
            HVX_سمتیہ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        إذا (nloe) {                                                                       \
            HVX_سمتیہ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((فراغ *) &vdst[i], nloe * elem_ناپ, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } طالما(0)

ساكن مضمن فراغ hvx_scale_وتھ_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_وتھ_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_وتھ_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_وتھ_f32_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_وتھ_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_وتھ_f32_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_وتھ_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    hvx_scale_وتھ_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_وتھ_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    إذا (((size_t) dst & 127) == 0) {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_وتھ_f32_aa(dst, src, n, scale, وتھ);
        } وإلا {
            hvx_scale_وتھ_f32_au(dst, src, n, scale, وتھ);
        }
    } وإلا {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_وتھ_f32_ua(dst, src, n, scale, وتھ);
        } وإلا {
            hvx_scale_وتھ_f32_uu(dst, src, n, scale, وتھ);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#تعریف_کرو hvx_scale_وتھ_f16_loop_body(dst_type, src_type, vec_store)                                                \
    افعل {                                                                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                                                                 \
        src_type * محدود vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_سمتیہ vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_سمتیہ vo = hvx_vec_splat_f32(وتھ);                                                                   \
                                                                                                                     \
        ثابت uint32_t nvec = n / VLEN_FP16;                                                                         \
        ثابت uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        لكل (; i < nvec; ++i) {                                                                                      \
            HVX_سمتیہ_Pair p = hvx_vec_f16_منزل_f32(vsrc[i]);                                                          \
            HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_منزل_f16(r0, r1);                                                                    \
        }                                                                                                            \
        إذا (nloe) {                                                                                                  \
            HVX_سمتیہ_Pair p = hvx_vec_f16_منزل_f32(vsrc[i]);                                                          \
            HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_سمتیہ v = hvx_vec_f32_منزل_f16(r0, r1);                                                               \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } طالما(0)

ساكن مضمن فراغ hvx_scale_وتھ_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_وتھ_f16_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_وتھ_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_وتھ_f16_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_scale_وتھ_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_وتھ_f16_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_وتھ_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    hvx_scale_وتھ_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_scale_وتھ_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح n, ثابت عائم scale, ثابت عائم وتھ) {
    إذا (((size_t) dst & 127) == 0) {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_وتھ_f16_aa(dst, src, n, scale, وتھ);
        } وإلا {
            hvx_scale_وتھ_f16_au(dst, src, n, scale, وتھ);
        }
    } وإلا {
        إذا (((size_t) src & 127) == 0) {
            hvx_scale_وتھ_f16_ua(dst, src, n, scale, وتھ);
        } وإلا {
            hvx_scale_وتھ_f16_uu(dst, src, n, scale, وتھ);
        }
    }
}

#جے_انت // HVX_SCALE_H
