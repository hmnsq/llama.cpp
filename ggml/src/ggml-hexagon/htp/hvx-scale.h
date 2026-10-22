#ifndef HVX_SCALE_H
#define HVX_SCALE_H

#شامل <assert.h>
#شامل <stddef.h>
#شامل <stdint.h>

#شامل "hvx-base.h"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    انجامبده {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ثابت uint32_t elem_size = اندازه(اعشاری);                                    \
        ثابت uint32_t epv = 128 / elem_size;                                        \
        ثابت uint32_t nvec = n / epv;                                               \
        ثابت uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        برای (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        اگر (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((تهی *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } وقتی(0)

ایستا درونخطی تهی hvx_scale_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ایستا درونخطی تهی hvx_scale_f32_au(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ایستا درونخطی تهی hvx_scale_f32_ua(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ایستا درونخطی تهی hvx_scale_f32_uu(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ایستا درونخطی تهی hvx_scale_f32(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale) {
    اگر (((size_t) dst & 127) == 0) {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } وگرنه {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } وگرنه {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } وگرنه {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                     \
    انجامبده {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src_type * restrict vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                         \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                        \
                                                                                          \
        ثابت uint32_t elem_size = اندازه(اعشاری);                                         \
        ثابت uint32_t epv = 128 / elem_size;                                             \
        ثابت uint32_t nvec = n / epv;                                                    \
        ثابت uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        برای (; i < nvec; ++i) {                                                           \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        اگر (nloe) {                                                                       \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((تهی *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } وقتی(0)

ایستا درونخطی تهی hvx_scale_offset_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ایستا درونخطی تهی hvx_scale_offset_f32_au(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ایستا درونخطی تهی hvx_scale_offset_f32_ua(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ایستا درونخطی تهی hvx_scale_offset_f32_uu(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ایستا درونخطی تهی hvx_scale_offset_f32(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    اگر (((size_t) dst & 127) == 0) {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } وگرنه {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } وگرنه {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } وگرنه {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#define hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)                                                \
    انجامبده {                                                                                                             \
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
        برای (; i < nvec; ++i) {                                                                                      \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                                                    \
        }                                                                                                            \
        اگر (nloe) {                                                                                                  \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                                               \
            vec_store((تهی *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } وقتی(0)

ایستا درونخطی تهی hvx_scale_offset_f16_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

ایستا درونخطی تهی hvx_scale_offset_f16_au(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

ایستا درونخطی تهی hvx_scale_offset_f16_ua(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

ایستا درونخطی تهی hvx_scale_offset_f16_uu(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ایستا درونخطی تهی hvx_scale_offset_f16(uint8_t * restrict dst, ثابت uint8_t * restrict src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری offset) {
    اگر (((size_t) dst & 127) == 0) {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_aa(dst, src, n, scale, offset);
        } وگرنه {
            hvx_scale_offset_f16_au(dst, src, n, scale, offset);
        }
    } وگرنه {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_ua(dst, src, n, scale, offset);
        } وگرنه {
            hvx_scale_offset_f16_uu(dst, src, n, scale, offset);
        }
    }
}

#endif // HVX_SCALE_H
