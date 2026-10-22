#جيڪڏهن_وصف_نه_ٿيل HVX_SQRT_H
#وصف_ڪريو HVX_SQRT_H

#ضمن <stdbool.h>
#ضمن <stdint.h>

#ضمن "hex-utils.h"

#ضمن "hvx-base.h"

#وصف_ڪريو RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#وصف_ڪريو RSQRT_ONE_HALF     0x3f000000  // 0.5
#وصف_ڪريو RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#إذا __HVX_ARCH__ < 79
#وصف_ڪريو HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#وإلا
#وصف_ڪريو HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#جيڪڏهن_پورو

ساكن مضمن HVX_ويڪٽر hvx_vec_rsqrt_f32(HVX_ويڪٽر اندريون_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_ويڪٽر rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_ويڪٽر onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_ويڪٽر threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_ويڪٽر x2, y, ypower2, عارضي;

    x2 = Q6_Vqf32_vmpy_VsfVsf(اندريون_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(اندريون_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    عارضي    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    عارضي    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(عارضي));
    عارضي    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(عارضي));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(عارضي, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    عارضي    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    عارضي    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(عارضي));
    عارضي    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, عارضي);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(عارضي, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    عارضي    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    عارضي    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(عارضي));
    عارضي    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, عارضي);

    أرجع Q6_Vsf_equals_Vqf32(عارضي);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#وصف_ڪريو hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
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
            HVX_ويڪٽر inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_ويڪٽر sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        إذا (nloe) {                                                          \
            HVX_ويڪٽر inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_ويڪٽر sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } طالما(0)

ساكن مضمن فراغ hvx_sqrt_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشان ڊگهو) dst % 128 == 0);
    أكد((بنانشان ڊگهو) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_ويڪٽر, HVX_ويڪٽر, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشان ڊگهو) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_ويڪٽر, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشان ڊگهو) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_ويڪٽر, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    إذا ((بنانشان ڊگهو) dst % 128 == 0) {
        إذا ((بنانشان ڊگهو) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, انگ_elems);
        } وإلا {
            hvx_sqrt_f32_au(dst, src, انگ_elems);
        }
    } وإلا {
        إذا ((بنانشان ڊگهو) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, انگ_elems);
        } وإلا {
            hvx_sqrt_f32_uu(dst, src, انگ_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#وصف_ڪريو hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
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
            HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(vsrc[i]);                            \
            HVX_ويڪٽر r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_ويڪٽر r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_منزل_f16(r0, r1);                                      \
        }                                                                              \
        إذا (nloe) {                                                                    \
            HVX_ويڪٽر_Pair p = hvx_vec_f16_منزل_f32(vsrc[i]);                            \
            HVX_ويڪٽر r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_ويڪٽر r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_ويڪٽر v = hvx_vec_f32_منزل_f16(r0, r1);                                 \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } طالما(0)

ساكن مضمن فراغ hvx_sqrt_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشان ڊگهو) dst % 128 == 0);
    أكد((بنانشان ڊگهو) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_ويڪٽر, HVX_ويڪٽر, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشان ڊگهو) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_ويڪٽر, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_sqrt_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشان ڊگهو) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_ويڪٽر, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_sqrt_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت صحيح انگ_elems) {
    إذا ((بنانشان ڊگهو) dst % 128 == 0) {
        إذا ((بنانشان ڊگهو) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, انگ_elems);
        } وإلا {
            hvx_sqrt_f16_au(dst, src, انگ_elems);
        }
    } وإلا {
        إذا ((بنانشان ڊگهو) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, انگ_elems);
        } وإلا {
            hvx_sqrt_f16_uu(dst, src, انگ_elems);
        }
    }
}

#جيڪڏهن_پورو /* HVX_SQRT_H */
