#ifndef HVX_SQRT_H
#define HVX_SQRT_H

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>

#அந்தர்பாவ "hex-utils.h"

#அந்தர்பாவ "hvx-base.h"

#define RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#define RSQRT_ONE_HALF     0x3f000000  // 0.5
#define RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#எனில் __HVX_ARCH__ < 79
#define HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#இல்லையெனில்
#define HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

நிலையான உள்வரி HVX_Vector hvx_vec_rsqrt_f32(HVX_Vector in_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_Vector rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_Vector x2, y, ypower2, temp;

    x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    திரும்பு Q6_Vsf_equals_Vqf32(temp);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#define hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    செய் {                                                                     \
        dst_type * restrict vdst = (dst_type *) dst;                         \
        src_type * restrict vsrc = (src_type *) src;                         \
                                                                             \
        மாறிலி uint32_t nvec = n / VLEN_FP32;                                 \
        மாறிலி uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        க்கு (; i < nvec; i++) {                                              \
            HVX_Vector inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_Vector sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        எனில் (nloe) {                                                          \
            HVX_Vector inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_Vector sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((வெறுமை *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } போது(0)

நிலையான உள்வரி வெறுமை hvx_sqrt_f32_aa(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    ப்ரதிஜ்ஞா((unsigned long) dst % 128 == 0);
    ப்ரதிஜ்ஞா((unsigned long) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f32_au(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    ப்ரதிஜ்ஞா((unsigned long) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f32_ua(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    ப்ரதிஜ்ஞா((unsigned long) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f32_uu(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f32(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    எனில் ((unsigned long) dst % 128 == 0) {
        எனில் ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, இலக்கம்_elems);
        } இல்லையெனில் {
            hvx_sqrt_f32_au(dst, src, இலக்கம்_elems);
        }
    } இல்லையெனில் {
        எனில் ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, இலக்கம்_elems);
        } இல்லையெனில் {
            hvx_sqrt_f32_uu(dst, src, இலக்கம்_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#define hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    செய் {                                                                               \
        dst_type * restrict vdst = (dst_type *) dst;                                   \
        src_type * restrict vsrc = (src_type *) src;                                   \
                                                                                       \
        மாறிலி uint32_t nvec = n / VLEN_FP16;                                           \
        மாறிலி uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        க்கு (; i < nvec; i++) {                                                        \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                            \
            HVX_Vector r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_Vector r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                      \
        }                                                                              \
        எனில் (nloe) {                                                                    \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                            \
            HVX_Vector r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_Vector r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                 \
            vec_store((வெறுமை *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } போது(0)

நிலையான உள்வரி வெறுமை hvx_sqrt_f16_aa(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    ப்ரதிஜ்ஞா((unsigned long) dst % 128 == 0);
    ப்ரதிஜ்ஞா((unsigned long) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f16_au(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    ப்ரதிஜ்ஞா((unsigned long) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f16_ua(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    ப்ரதிஜ்ஞா((unsigned long) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f16_uu(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

நிலையான உள்வரி வெறுமை hvx_sqrt_f16(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி முழு இலக்கம்_elems) {
    எனில் ((unsigned long) dst % 128 == 0) {
        எனில் ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, இலக்கம்_elems);
        } இல்லையெனில் {
            hvx_sqrt_f16_au(dst, src, இலக்கம்_elems);
        }
    } இல்லையெனில் {
        எனில் ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, இலக்கம்_elems);
        } இல்லையெனில் {
            hvx_sqrt_f16_uu(dst, src, இலக்கம்_elems);
        }
    }
}

#endif /* HVX_SQRT_H */
