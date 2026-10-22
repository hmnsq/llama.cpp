#ifndef HVX_SQRT_H
#define HVX_SQRT_H

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব "hex-utils.h"

#অন্তর্ভাব "hvx-base.h"

#define RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#define RSQRT_ONE_HALF     0x3f000000  // 0.5
#define RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#যদি __HVX_ARCH__ < 79
#define HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#অন্যথা
#define HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

স্থির অন্তঃরেখ HVX_Vector hvx_vec_rsqrt_f32(HVX_Vector in_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_Vector rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_Vector x2, y, ypower2, অস্থায়ী;

    x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    অস্থায়ী    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    অস্থায়ী    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(অস্থায়ী));
    অস্থায়ী    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(অস্থায়ী));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(অস্থায়ী, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    অস্থায়ী    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    অস্থায়ী    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(অস্থায়ী));
    অস্থায়ী    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, অস্থায়ী);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(অস্থায়ী, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    অস্থায়ী    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    অস্থায়ী    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(অস্থায়ী));
    অস্থায়ী    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, অস্থায়ী);

    ফেরত Q6_Vsf_equals_Vqf32(অস্থায়ী);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#define hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    করো {                                                                     \
        dst_type * restrict vdst = (dst_type *) dst;                         \
        src_type * restrict vsrc = (src_type *) src;                         \
                                                                             \
        ধ্রুবক uint32_t nvec = n / VLEN_FP32;                                 \
        ধ্রুবক uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        জন্য (; i < nvec; i++) {                                              \
            HVX_Vector inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_Vector sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        যদি (nloe) {                                                          \
            HVX_Vector inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_Vector sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((শূন্য *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f32_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f32_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f32_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি ((unsigned long) dst % 128 == 0) {
        যদি ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqrt_f32_au(dst, src, অঙ্ক_elems);
        }
    } অন্যথা {
        যদি ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqrt_f32_uu(dst, src, অঙ্ক_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#define hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    করো {                                                                               \
        dst_type * restrict vdst = (dst_type *) dst;                                   \
        src_type * restrict vsrc = (src_type *) src;                                   \
                                                                                       \
        ধ্রুবক uint32_t nvec = n / VLEN_FP16;                                           \
        ধ্রুবক uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        জন্য (; i < nvec; i++) {                                                        \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                            \
            HVX_Vector r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_Vector r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                      \
        }                                                                              \
        যদি (nloe) {                                                                    \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                            \
            HVX_Vector r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_Vector r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                 \
            vec_store((শূন্য *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } যখন(0)

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f16_au(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f16_ua(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f16_uu(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

স্থির অন্তঃরেখ শূন্য hvx_sqrt_f16(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    যদি ((unsigned long) dst % 128 == 0) {
        যদি ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqrt_f16_au(dst, src, অঙ্ক_elems);
        }
    } অন্যথা {
        যদি ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, অঙ্ক_elems);
        } অন্যথা {
            hvx_sqrt_f16_uu(dst, src, অঙ্ক_elems);
        }
    }
}

#endif /* HVX_SQRT_H */
