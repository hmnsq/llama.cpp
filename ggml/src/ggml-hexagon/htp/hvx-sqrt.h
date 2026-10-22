#నిర్వచితం_కాకపోతే HVX_SQRT_H
#నిర్వచించు HVX_SQRT_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hex-utils.h"

#అన్తర్భావ "hvx-base.h"

#నిర్వచించు RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#నిర్వచించు RSQRT_ONE_HALF     0x3f000000  // 0.5
#నిర్వచించు RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#ఒకవేళ __HVX_ARCH__ < 79
#నిర్వచించు HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#లేకపోతే
#నిర్వచించు HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#అయితే_ముగింపు

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_rsqrt_f32(HVX_సదిశ లోపల_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_సదిశ rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_సదిశ onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_సదిశ threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_సదిశ x2, y, ypower2, తాత్కాలిక;

    x2 = Q6_Vqf32_vmpy_VsfVsf(లోపల_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(లోపల_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    తాత్కాలిక    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    తాత్కాలిక    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(తాత్కాలిక));
    తాత్కాలిక    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(తాత్కాలిక));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(తాత్కాలిక, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    తాత్కాలిక    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    తాత్కాలిక    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(తాత్కాలిక));
    తాత్కాలిక    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, తాత్కాలిక);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(తాత్కాలిక, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    తాత్కాలిక    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    తాత్కాలిక    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(తాత్కాలిక));
    తాత్కాలిక    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, తాత్కాలిక);

    తిరిగి Q6_Vsf_equals_Vqf32(తాత్కాలిక);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#నిర్వచించు hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    చేయి {                                                                     \
        dst_type * పరిమిత vdst = (dst_type *) dst;                         \
        src_type * పరిమిత vsrc = (src_type *) src;                         \
                                                                             \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP32;                                 \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        కొరకు (; i < nvec; i++) {                                              \
            HVX_సదిశ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_సదిశ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        ఒకవేళ (nloe) {                                                          \
            HVX_సదిశ inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_సదిశ sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ ((గుర్తులేని దీర్ఘ) dst % 128 == 0) {
        ఒకవేళ ((గుర్తులేని దీర్ఘ) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqrt_f32_au(dst, src, అంకె_elems);
        }
    } లేకపోతే {
        ఒకవేళ ((గుర్తులేని దీర్ఘ) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqrt_f32_uu(dst, src, అంకె_elems);
        }
    }
}

// Compute sqrt(x) for f16 by promoting to f32, applying hvx_vec_rsqrt_f32, and narrowing back.
#నిర్వచించు hvx_sqrt_f16_loop_body(dst_type, src_type, vec_store)                          \
    చేయి {                                                                               \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                   \
        src_type * పరిమిత vsrc = (src_type *) src;                                   \
                                                                                       \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP16;                                           \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP16;                                           \
                                                                                       \
        uint32_t i = 0;                                                                \
                                                                                       \
        _Pragma("unroll(4)")                                                           \
        కొరకు (; i < nvec; i++) {                                                        \
            HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(vsrc[i]);                            \
            HVX_సదిశ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_సదిశ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            vdst[i] = hvx_vec_f32_గమ్యం_f16(r0, r1);                                      \
        }                                                                              \
        ఒకవేళ (nloe) {                                                                    \
            HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(vsrc[i]);                            \
            HVX_సదిశ r0 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_lo_W(p)), Q6_V_lo_W(p)); \
            HVX_సదిశ r1 = HVX_OP_MUL(hvx_vec_rsqrt_f32(Q6_V_hi_W(p)), Q6_V_hi_W(p)); \
            HVX_సదిశ v = hvx_vec_f32_గమ్యం_f16(r0, r1);                                 \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP16, v);                       \
        }                                                                              \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_sqrt_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqrt_f16(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ ((గుర్తులేని దీర్ఘ) dst % 128 == 0) {
        ఒకవేళ ((గుర్తులేని దీర్ఘ) src % 128 == 0) {
            hvx_sqrt_f16_aa(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqrt_f16_au(dst, src, అంకె_elems);
        }
    } లేకపోతే {
        ఒకవేళ ((గుర్తులేని దీర్ఘ) src % 128 == 0) {
            hvx_sqrt_f16_ua(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqrt_f16_uu(dst, src, అంకె_elems);
        }
    }
}

#అయితే_ముగింపు /* HVX_SQRT_H */
