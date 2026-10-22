#اگر_تعریف_نشده HVX_SQRT_H
#تعریف_کن HVX_SQRT_H

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن "hex-utils.h"

#ضمن "hvx-base.h"

#تعریف_کن RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#تعریف_کن RSQRT_ONE_HALF     0x3f000000  // 0.5
#تعریف_کن RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#إذا __HVX_ARCH__ < 79
#تعریف_کن HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#وإلا
#تعریف_کن HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#پایان_اگر

ساكن مضمن HVX_وکتور hvx_vec_rsqrt_f32(HVX_وکتور درون_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_وکتور rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_وکتور onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_وکتور threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_وکتور x2, y, ypower2, موقت;

    x2 = Q6_Vqf32_vmpy_VsfVsf(درون_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(درون_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    موقت    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    موقت    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(موقت));
    موقت    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(موقت));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(موقت, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    موقت    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    موقت    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(موقت));
    موقت    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, موقت);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(موقت, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    موقت    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    موقت    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(موقت));
    موقت    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, موقت);

    أرجع Q6_Vsf_equals_Vqf32(موقت);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#تعریف_کن hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    افعل {                                                                     \
        dst_type * محدود vdst = (dst_type *) dst;                         \
        src_type * محدود vsrc = (src_type *) src;                         \
                                                                             \
        ثابت uint32_t nvec = n / VLEN_FP32;                                 \
        ثابت uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        لكل (; i < nvec; i++) {                                              \
            HVX_وکتور inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_وکتور sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        إذا (nloe) {                                                          \
            HVX_وکتور inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_وکتور sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } طالما(0)

ساكن مضمن فراغ hvx_sqrt_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_وکتور, HVX_وکتور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_وکتور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_وکتور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    إذا ((بیعلامت دراز) dst % 128 == 0) {
        إذا ((بیعلامت دراز) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, رقم_elems);
        } وإلا {
            hvx_sqrt_f32_au(dst, src, رقم_elems);
        }
    } وإلا {
        إذا ((بیعلامت دراز) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, رقم_elems);
        } وإلا {
            hvx_sqrt_f32_uu(dst, src, رقم_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#تعریف_کن hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    افعل {                                                                               \
        dst_type * محدود vdst = (dst_type *) dst;                                   \
        src_type * محدود vsrc = (src_type *) src;                                   \
                                                                                       \
        ثابت uint32_t nvec = n / VLEN_FP16;                                           \
        ثابت uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        لكل (; i < nvec; i++) {                                                        \
            HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(vsrc[i]);                            \
            HVX_وکتور r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_وکتور r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_مقصد_f16(r0, r1);                                      \
        }                                                                              \
        إذا (nloe) {                                                                    \
            HVX_وکتور_Pair p = hvx_vec_f16_مقصد_f32(vsrc[i]);                            \
            HVX_وکتور r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_وکتور r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_وکتور v = hvx_vec_f32_مقصد_f16(r0, r1);                                 \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } طالما(0)

ساكن مضمن فراغ hvx_sqrt_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_وکتور, HVX_وکتور, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_وکتور, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_وکتور, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح رقم_elems) {
    إذا ((بیعلامت دراز) dst % 128 == 0) {
        إذا ((بیعلامت دراز) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, رقم_elems);
        } وإلا {
            hvx_sqrt_f16_au(dst, src, رقم_elems);
        }
    } وإلا {
        إذا ((بیعلامت دراز) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, رقم_elems);
        } وإلا {
            hvx_sqrt_f16_uu(dst, src, رقم_elems);
        }
    }
}

#پایان_اگر /* HVX_SQRT_H */
