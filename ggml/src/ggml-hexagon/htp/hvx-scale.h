#اگر_غیر_تعریف_شدہ HVX_SCALE_H
#تعریف_کرو HVX_SCALE_H

#شامل <assert.h>
#شامل <stddef.h>
#شامل <stdint.h>

#شامل "hvx-base.h"

#تعریف_کرو hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    کرو {                                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                                 \
        src_type * محدود vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_سمتیہ vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        ثابت uint32_t elem_سائز = حجم(اعشاری);                                    \
        ثابت uint32_t epv = 128 / elem_سائز;                                        \
        ثابت uint32_t nvec = n / epv;                                               \
        ثابت uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        برائے (; i < nvec; ++i) {                                                      \
            HVX_سمتیہ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        اگر (nloe) {                                                                  \
            HVX_سمتیہ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((خالی *) &vdst[i], nloe * elem_سائز, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } جب(0)

ساکن اندرونی خالی hvx_scale_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_scale_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_scale_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_scale_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_scale_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale) {
    اگر (((size_t) dst & 127) == 0) {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } ورنہ {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } ورنہ {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } ورنہ {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#تعریف_کرو hvx_scale_انحراف_f32_loop_body(dst_type, src_type, vec_store)                     \
    کرو {                                                                                  \
        dst_type * محدود vdst = (dst_type *) dst;                                      \
        src_type * محدود vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_سمتیہ vs = hvx_vec_splat_f32(scale);                                         \
        HVX_سمتیہ vo = hvx_vec_splat_f32(انحراف);                                        \
                                                                                          \
        ثابت uint32_t elem_سائز = حجم(اعشاری);                                         \
        ثابت uint32_t epv = 128 / elem_سائز;                                             \
        ثابت uint32_t nvec = n / epv;                                                    \
        ثابت uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        برائے (; i < nvec; ++i) {                                                           \
            HVX_سمتیہ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        اگر (nloe) {                                                                       \
            HVX_سمتیہ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((خالی *) &vdst[i], nloe * elem_سائز, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } جب(0)

ساکن اندرونی خالی hvx_scale_انحراف_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_انحراف_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_scale_انحراف_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_انحراف_f32_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_scale_انحراف_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_انحراف_f32_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_scale_انحراف_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    hvx_scale_انحراف_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_scale_انحراف_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    اگر (((size_t) dst & 127) == 0) {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_انحراف_f32_aa(dst, src, n, scale, انحراف);
        } ورنہ {
            hvx_scale_انحراف_f32_au(dst, src, n, scale, انحراف);
        }
    } ورنہ {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_انحراف_f32_ua(dst, src, n, scale, انحراف);
        } ورنہ {
            hvx_scale_انحراف_f32_uu(dst, src, n, scale, انحراف);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#تعریف_کرو hvx_scale_انحراف_f16_loop_body(dst_type, src_type, vec_store)                                                \
    کرو {                                                                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                                                                 \
        src_type * محدود vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_سمتیہ vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_سمتیہ vo = hvx_vec_splat_f32(انحراف);                                                                   \
                                                                                                                     \
        ثابت uint32_t nvec = n / VLEN_FP16;                                                                         \
        ثابت uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        برائے (; i < nvec; ++i) {                                                                                      \
            HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(vsrc[i]);                                                          \
            HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_مقصد_f16(r0, r1);                                                                    \
        }                                                                                                            \
        اگر (nloe) {                                                                                                  \
            HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(vsrc[i]);                                                          \
            HVX_سمتیہ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_سمتیہ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_سمتیہ v = hvx_vec_f32_مقصد_f16(r0, r1);                                                               \
            vec_store((خالی *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } جب(0)

ساکن اندرونی خالی hvx_scale_انحراف_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);
    hvx_scale_انحراف_f16_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_scale_انحراف_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    أكد((size_t) dst % 128 == 0);
    hvx_scale_انحراف_f16_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_scale_انحراف_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    أكد((size_t) src % 128 == 0);
    hvx_scale_انحراف_f16_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_scale_انحراف_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    hvx_scale_انحراف_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_scale_انحراف_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح n, ثابت اعشاری scale, ثابت اعشاری انحراف) {
    اگر (((size_t) dst & 127) == 0) {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_انحراف_f16_aa(dst, src, n, scale, انحراف);
        } ورنہ {
            hvx_scale_انحراف_f16_au(dst, src, n, scale, انحراف);
        }
    } ورنہ {
        اگر (((size_t) src & 127) == 0) {
            hvx_scale_انحراف_f16_ua(dst, src, n, scale, انحراف);
        } ورنہ {
            hvx_scale_انحراف_f16_uu(dst, src, n, scale, انحراف);
        }
    }
}

#اگر_ختم // HVX_SCALE_H
