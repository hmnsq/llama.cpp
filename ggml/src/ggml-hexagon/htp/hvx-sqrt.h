#اگر_غیر_تعریف_شدہ HVX_SQRT_H
#تعریف_کرو HVX_SQRT_H

#شامل <stdbool.h>
#شامل <stdint.h>

#شامل "hex-utils.h"

#شامل "hvx-base.h"

#تعریف_کرو RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#تعریف_کرو RSQRT_ONE_HALF     0x3f000000  // 0.5
#تعریف_کرو RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#اگر __HVX_ARCH__ < 79
#تعریف_کرو HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ورنہ
#تعریف_کرو HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#اگر_ختم

ساکن اندرونی HVX_سمتیہ hvx_vec_rsqrt_f32(HVX_سمتیہ اندر_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_سمتیہ rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_سمتیہ onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_سمتیہ threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_سمتیہ x2, y, ypower2, عارضی;

    x2 = Q6_Vqf32_vmpy_VsfVsf(اندر_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(اندر_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    عارضی    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    عارضی    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(عارضی));
    عارضی    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(عارضی));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(عارضی, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    عارضی    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    عارضی    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(عارضی));
    عارضی    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, عارضی);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(عارضی, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    عارضی    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    عارضی    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(عارضی));
    عارضی    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, عارضی);

    واپس Q6_Vsf_equals_Vqf32(عارضی);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#تعریف_کرو hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    کرو {                                                                     \
        dst_type * محدود vdst = (dst_type *) dst;                         \
        src_type * محدود vsrc = (src_type *) src;                         \
                                                                             \
        ثابت uint32_t nvec = n / VLEN_FP32;                                 \
        ثابت uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        برائے (; i < nvec; i++) {                                              \
            HVX_سمتیہ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_سمتیہ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        اگر (nloe) {                                                          \
            HVX_سمتیہ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_سمتیہ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((خالی *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } جب(0)

ساکن اندرونی خالی hvx_sqrt_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_sqrt_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_sqrt_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_sqrt_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_sqrt_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    اگر ((غیرعلامتی طویل) dst % 128 == 0) {
        اگر ((غیرعلامتی طویل) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, عدد_elems);
        } ورنہ {
            hvx_sqrt_f32_au(dst, src, عدد_elems);
        }
    } ورنہ {
        اگر ((غیرعلامتی طویل) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, عدد_elems);
        } ورنہ {
            hvx_sqrt_f32_uu(dst, src, عدد_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#تعریف_کرو hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    کرو {                                                                               \
        dst_type * محدود vdst = (dst_type *) dst;                                   \
        src_type * محدود vsrc = (src_type *) src;                                   \
                                                                                       \
        ثابت uint32_t nvec = n / VLEN_FP16;                                           \
        ثابت uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        برائے (; i < nvec; i++) {                                                        \
            HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(vsrc[i]);                            \
            HVX_سمتیہ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_سمتیہ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_مقصد_f16(r0, r1);                                      \
        }                                                                              \
        اگر (nloe) {                                                                    \
            HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(vsrc[i]);                            \
            HVX_سمتیہ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_سمتیہ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_سمتیہ v = hvx_vec_f32_مقصد_f16(r0, r1);                                 \
            vec_store((خالی *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } جب(0)

ساکن اندرونی خالی hvx_sqrt_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_sqrt_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_sqrt_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_sqrt_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_sqrt_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحیح عدد_elems) {
    اگر ((غیرعلامتی طویل) dst % 128 == 0) {
        اگر ((غیرعلامتی طویل) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, عدد_elems);
        } ورنہ {
            hvx_sqrt_f16_au(dst, src, عدد_elems);
        }
    } ورنہ {
        اگر ((غیرعلامتی طویل) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, عدد_elems);
        } ورنہ {
            hvx_sqrt_f16_uu(dst, src, عدد_elems);
        }
    }
}

#اگر_ختم /* HVX_SQRT_H */
