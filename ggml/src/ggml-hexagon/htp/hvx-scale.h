#నిర్వచితం_కాకపోతే HVX_SCALE_H
#నిర్వచించు HVX_SCALE_H

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hvx-base.h"

#నిర్వచించు hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    చేయి {                                                                             \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                 \
        src_type * పరిమిత vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_సదిశ vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);                                    \
        స్థిరరాశి uint32_t epv = 128 / elem_కొలత;                                        \
        స్థిరరాశి uint32_t nvec = n / epv;                                               \
        స్థిరరాశి uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        కొరకు (; i < nvec; ++i) {                                                      \
            HVX_సదిశ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        ఒకవేళ (nloe) {                                                                  \
            HVX_సదిశ v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_scale_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale) {
    ప్రతిజ్ఞా((size_t) dst % 128 == 0);
    ప్రతిజ్ఞా((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale) {
    ప్రతిజ్ఞా((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale) {
    ప్రతిజ్ఞా((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale) {
    ఒకవేళ (((size_t) dst & 127) == 0) {
        ఒకవేళ (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } లేకపోతే {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } లేకపోతే {
        ఒకవేళ (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } లేకపోతే {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#నిర్వచించు hvx_scale_స్థానభ్రంశం_f32_loop_body(dst_type, src_type, vec_store)                     \
    చేయి {                                                                                  \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                      \
        src_type * పరిమిత vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_సదిశ vs = hvx_vec_splat_f32(scale);                                         \
        HVX_సదిశ vo = hvx_vec_splat_f32(స్థానభ్రంశం);                                        \
                                                                                          \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);                                         \
        స్థిరరాశి uint32_t epv = 128 / elem_కొలత;                                             \
        స్థిరరాశి uint32_t nvec = n / epv;                                                    \
        స్థిరరాశి uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        కొరకు (; i < nvec; ++i) {                                                           \
            HVX_సదిశ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        ఒకవేళ (nloe) {                                                                       \
            HVX_సదిశ v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ప్రతిజ్ఞా((size_t) dst % 128 == 0);
    ప్రతిజ్ఞా((size_t) src % 128 == 0);
    hvx_scale_స్థానభ్రంశం_f32_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ప్రతిజ్ఞా((size_t) dst % 128 == 0);
    hvx_scale_స్థానభ్రంశం_f32_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ప్రతిజ్ఞా((size_t) src % 128 == 0);
    hvx_scale_స్థానభ్రంశం_f32_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    hvx_scale_స్థానభ్రంశం_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ఒకవేళ (((size_t) dst & 127) == 0) {
        ఒకవేళ (((size_t) src & 127) == 0) {
            hvx_scale_స్థానభ్రంశం_f32_aa(dst, src, n, scale, స్థానభ్రంశం);
        } లేకపోతే {
            hvx_scale_స్థానభ్రంశం_f32_au(dst, src, n, scale, స్థానభ్రంశం);
        }
    } లేకపోతే {
        ఒకవేళ (((size_t) src & 127) == 0) {
            hvx_scale_స్థానభ్రంశం_f32_ua(dst, src, n, scale, స్థానభ్రంశం);
        } లేకపోతే {
            hvx_scale_స్థానభ్రంశం_f32_uu(dst, src, n, scale, స్థానభ్రంశం);
        }
    }
}

// Scale+offset computed by promoting f16 -> f32, then narrowing the result back to f16.
#నిర్వచించు hvx_scale_స్థానభ్రంశం_f16_loop_body(dst_type, src_type, vec_store)                                                \
    చేయి {                                                                                                             \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                                                 \
        src_type * పరిమిత vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_సదిశ vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_సదిశ vo = hvx_vec_splat_f32(స్థానభ్రంశం);                                                                   \
                                                                                                                     \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP16;                                                                         \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        కొరకు (; i < nvec; ++i) {                                                                                      \
            HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(vsrc[i]);                                                          \
            HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            vdst[i] = hvx_vec_f32_గమ్యం_f16(r0, r1);                                                                    \
        }                                                                                                            \
        ఒకవేళ (nloe) {                                                                                                  \
            HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(vsrc[i]);                                                          \
            HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo)); \
            HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo)); \
            HVX_సదిశ v = hvx_vec_f32_గమ్యం_f16(r0, r1);                                                               \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ప్రతిజ్ఞా((size_t) dst % 128 == 0);
    ప్రతిజ్ఞా((size_t) src % 128 == 0);
    hvx_scale_స్థానభ్రంశం_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ప్రతిజ్ఞా((size_t) dst % 128 == 0);
    hvx_scale_స్థానభ్రంశం_f16_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ప్రతిజ్ఞా((size_t) src % 128 == 0);
    hvx_scale_స్థానభ్రంశం_f16_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    hvx_scale_స్థానభ్రంశం_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_scale_స్థానభ్రంశం_f16(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం స్థానభ్రంశం) {
    ఒకవేళ (((size_t) dst & 127) == 0) {
        ఒకవేళ (((size_t) src & 127) == 0) {
            hvx_scale_స్థానభ్రంశం_f16_aa(dst, src, n, scale, స్థానభ్రంశం);
        } లేకపోతే {
            hvx_scale_స్థానభ్రంశం_f16_au(dst, src, n, scale, స్థానభ్రంశం);
        }
    } లేకపోతే {
        ఒకవేళ (((size_t) src & 127) == 0) {
            hvx_scale_స్థానభ్రంశం_f16_ua(dst, src, n, scale, స్థానభ్రంశం);
        } లేకపోతే {
            hvx_scale_స్థానభ్రంశం_f16_uu(dst, src, n, scale, స్థానభ్రంశం);
        }
    }
}

#అయితే_ముగింపు // HVX_SCALE_H
