#నిర్వచితం_కాకపోతే HVX_ARITH_H
#నిర్వచించు HVX_ARITH_H

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <math.h>

#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hex-utils.h"

//
// Binary operations (add, mul, sub)
//

#నిర్వచించు వాడని(x) (శూన్య)(x)

#నిర్వచించు hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_కొలత, vec_store, vec_op) \
    చేయి {                                                                                  \
        dst_type * vdst  = (dst_type *) dst;                                              \
        src0_type * vsrc0 = (src0_type *) src0;                                           \
        src1_type * vsrc1 = (src1_type *) src1;                                           \
                                                                                          \
        స్థిరరాశి uint32_t epv  = 128 / (elem_కొలత);                                          \
        స్థిరరాశి uint32_t nvec = n / epv;                                                    \
        స్థిరరాశి uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        కొరకు (; i < nvec; i++) {                                                           \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                                         \
        }                                                                                 \
        ఒకవేళ (nloe) {                                                                       \
            HVX_సదిశ v = vec_op(vsrc0[i], vsrc1[i]);                                    \
            vec_store((శూన్య *) &vdst[i], nloe * (elem_కొలత), v);                          \
        }                                                                                 \
    } అయితే(0)

#ఒకవేళ __HVX_ARCH__ < 79

#నిర్వచించు HVX_OP_కూడు_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#నిర్వచించు HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#లేకపోతే

#నిర్వచించు HVX_OP_కూడు_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#నిర్వచించు HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#అయితే_ముగింపు

#నిర్వచించు HVX_OP_కూడు_F16(a, b) hvx_vec_కూడు_f16_f16(a, b)
#నిర్వచించు HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#నిర్వచించు HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#నిర్వచించు DEFINE_HVX_ద్విమాన_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE)                                           \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aaa(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                       \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                      \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_సదిశ, HVX_సదిశ, HVX_సదిశ, పరిమాణం(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);    \
}                                                                                                             \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aau(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                       \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_సదిశ, HVX_సదిశ, HVX_UVector, పరిమాణం(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aua(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                       \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_సదిశ, HVX_UVector, HVX_సదిశ, పరిమాణం(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);   \
}                                                                                                             \
స్థిర అంతఃవరుస శూన్య OP_NAME##_auu(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                       \
    hvx_arith_loop_body(HVX_సదిశ, HVX_UVector, HVX_UVector, పరిమాణం(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);  \
}                                                                                                             \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uaa(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                      \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_సదిశ, HVX_సదిశ, పరిమాణం(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);   \
}                                                                                                             \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uau(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_సదిశ, HVX_UVector, పరిమాణం(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uua(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                      \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_సదిశ, పరిమాణం(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);  \
}                                                                                                             \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uuu(uint8_t * dst, స్థిరరాశి uint8_t * src0, స్థిరరాశి uint8_t * src1, uint32_t n) {     \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, పరిమాణం(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
}                                                                                                             \

DEFINE_HVX_ద్విమాన_OP_VARIANTS(hvx_కూడు_f32, HVX_OP_కూడు_F32, దశాంశం)
DEFINE_HVX_ద్విమాన_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, దశాంశం)
DEFINE_HVX_ద్విమాన_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, దశాంశం)

DEFINE_HVX_ద్విమాన_OP_VARIANTS(hvx_కూడు_f16, HVX_OP_కూడు_F16, _Float16)
DEFINE_HVX_ద్విమాన_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_ద్విమాన_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#నిర్వచించు HVX_ద్విమాన_DISPATCHER(OP_NAME)                                                                                                       \
స్థిర అంతఃవరుస శూన్య OP_NAME(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, స్థిరరాశి uint32_t అంకె_elems) { \
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {                                                                                                 \
        ఒకవేళ (hex_is_aligned((శూన్య *) src0, 128)) {                                                                                            \
            ఒకవేళ (hex_is_aligned((శూన్య *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, అంకె_elems);                                               \
            లేకపోతే                                    OP_NAME##_aau(dst, src0, src1, అంకె_elems);                                               \
        } లేకపోతే {                                                                                                                             \
            ఒకవేళ (hex_is_aligned((శూన్య *) src1, 128)) OP_NAME##_aua(dst, src0, src1, అంకె_elems);                                               \
            లేకపోతే                                    OP_NAME##_auu(dst, src0, src1, అంకె_elems);                                               \
        }                                                                                                                                    \
    } లేకపోతే {                                                                                                                                 \
        ఒకవేళ (hex_is_aligned((శూన్య *) src0, 128)) {                                                                                            \
            ఒకవేళ (hex_is_aligned((శూన్య *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, అంకె_elems);                                               \
            లేకపోతే                                    OP_NAME##_uau(dst, src0, src1, అంకె_elems);                                               \
        } లేకపోతే {                                                                                                                             \
            ఒకవేళ (hex_is_aligned((శూన్య *) src1, 128)) OP_NAME##_uua(dst, src0, src1, అంకె_elems);                                               \
            లేకపోతే                                    OP_NAME##_uuu(dst, src0, src1, అంకె_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

HVX_ద్విమాన_DISPATCHER(hvx_కూడు_f32)
HVX_ద్విమాన_DISPATCHER(hvx_sub_f32)
HVX_ద్విమాన_DISPATCHER(hvx_mul_f32)

HVX_ద్విమాన_DISPATCHER(hvx_కూడు_f16)
HVX_ద్విమాన_DISPATCHER(hvx_sub_f16)
HVX_ద్విమాన_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
స్థిర అంతఃవరుస శూన్య hvx_mul_mul_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, స్థిరరాశి uint8_t * పరిమిత src2, స్థిరరాశి uint32_t అంకె_elems) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src0 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src1 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src2 % 128 == 0);

    HVX_సదిశ * పరిమిత vdst  = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc0 = (HVX_సదిశ *) src0;
    HVX_సదిశ * పరిమిత vsrc1 = (HVX_సదిశ *) src1;
    HVX_సదిశ * పరిమిత vsrc2 = (HVX_సదిశ *) src2;

    స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;
    స్థిరరాశి uint32_t nvec = అంకె_elems / epv;
    స్థిరరాశి uint32_t nloe = అంకె_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        HVX_సదిశ v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_సదిశ v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * elem_కొలత, v2);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_mul_mul_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, స్థిరరాశి uint8_t * పరిమిత src2, స్థిరరాశి uint32_t అంకె_elems) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src0 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src1 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src2 % 128 == 0);

    HVX_సదిశ * పరిమిత vdst  = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc0 = (HVX_సదిశ *) src0;
    HVX_సదిశ * పరిమిత vsrc1 = (HVX_సదిశ *) src1;
    HVX_సదిశ * పరిమిత vsrc2 = (HVX_సదిశ *) src2;

    స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(_Float16);
    స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;
    స్థిరరాశి uint32_t nvec = అంకె_elems / epv;
    స్థిరరాశి uint32_t nloe = అంకె_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        HVX_సదిశ v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL_F16(v1, vsrc2[i]);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ v1 = HVX_OP_MUL_F16(vsrc0[i], vsrc1[i]);
        HVX_సదిశ v2 = HVX_OP_MUL_F16(v1, vsrc2[i]);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * elem_కొలత, v2);
    }
}

// Scalar Operations

#నిర్వచించు hvx_scalar_loop_body(dst_type, src_type, elem_కొలత, vec_store, scalar_op_macro) \
    చేయి {                                                                                \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                    \
        src_type * పరిమిత vsrc = (src_type *) src;                                    \
                                                                                        \
        స్థిరరాశి uint32_t epv  = 128 / (elem_కొలత);                                        \
        స్థిరరాశి uint32_t nvec = n / epv;                                                  \
        స్థిరరాశి uint32_t nloe = n % epv;                                                  \
                                                                                        \
        uint32_t i = 0;                                                                 \
                                                                                        \
        _Pragma("unroll(4)")                                                            \
        కొరకు (; i < nvec; i++) {                                                         \
            HVX_సదిశ v = vsrc[i];                                                     \
            vdst[i] = scalar_op_macro(v);                                               \
        }                                                                               \
        ఒకవేళ (nloe) {                                                                     \
            HVX_సదిశ v = vsrc[i];                                                     \
            v = scalar_op_macro(v);                                                     \
            vec_store((శూన్య *) &vdst[i], nloe * (elem_కొలత), v);                        \
        }                                                                               \
    } అయితే(0)

#నిర్వచించు HVX_OP_కూడు_SCALAR_F32(v)                                   \
    ({                                                             \
        స్థిరరాశి HVX_సదిశ_Pred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_సదిశ బహిర్గతం = HVX_OP_కూడు_F32(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, బహిర్గతం);                         \
    })

#నిర్వచించు HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#నిర్వచించు HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#నిర్వచించు HVX_OP_కూడు_SCALAR_F16(v)                                   \
    ({                                                             \
        స్థిరరాశి HVX_సదిశ_Pred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_సదిశ బహిర్గతం = HVX_OP_కూడు_F16(v, val_vec);               \
        Q6_V_vmux_QVV(pred_inf, inf, బహిర్గతం);                         \
    })

#నిర్వచించు HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#నిర్వచించు HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#నిర్వచించు DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE)                                  \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి ELEM_TYPE val, uint32_t n) { \
    స్థిరరాశి HVX_సదిశ val_vec = SPLAT_MACRO(val);                                                                         \
    స్థిరరాశి HVX_సదిశ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); వాడని(inf);                                                \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                                  \
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_సదిశ, HVX_సదిశ, పరిమాణం(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                          \
}                                                                                                                        \
స్థిర అంతఃవరుస శూన్య OP_NAME##_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి ELEM_TYPE val, uint32_t n) { \
    స్థిరరాశి HVX_సదిశ val_vec = SPLAT_MACRO(val);                                                                         \
    స్థిరరాశి HVX_సదిశ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); వాడని(inf);                                                \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_సదిశ, HVX_UVector, పరిమాణం(ELEM_TYPE), hvx_vec_store_a, OP_MACRO);                         \
}                                                                                                                        \
స్థిర అంతఃవరుస శూన్య OP_NAME##_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి ELEM_TYPE val, uint32_t n) { \
    స్థిరరాశి HVX_సదిశ val_vec = SPLAT_MACRO(val);                                                                         \
    స్థిరరాశి HVX_సదిశ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); వాడని(inf);                                                \
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);                                                                                  \
    hvx_scalar_loop_body(HVX_UVector, HVX_సదిశ, పరిమాణం(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                         \
}                                                                                                                        \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి ELEM_TYPE val, uint32_t n) { \
    స్థిరరాశి HVX_సదిశ val_vec = SPLAT_MACRO(val);                                                                         \
    స్థిరరాశి HVX_సదిశ inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); వాడని(inf);                                                \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, పరిమాణం(ELEM_TYPE), hvx_vec_store_u, OP_MACRO);                        \
}                                                                                                                        \

DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS(hvx_కూడు_scalar_f32, HVX_OP_కూడు_SCALAR_F32, hvx_vec_splat_f32, దశాంశం)
DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, దశాంశం)
DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, దశాంశం)

DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS(hvx_కూడు_scalar_f16, HVX_OP_కూడు_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#నిర్వచించు HVX_ద్విమాన_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE)                                                                          \
స్థిర అంతఃవరుస శూన్య OP_NAME(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి ELEM_TYPE val, స్థిరరాశి uint32_t అంకె_elems) { \
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {                                                 \
        OP_NAME##_aa(dst, src, val, అంకె_elems);                                                                                   \
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {                                                                               \
        OP_NAME##_au(dst, src, val, అంకె_elems);                                                                                   \
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {                                                                               \
        OP_NAME##_ua(dst, src, val, అంకె_elems);                                                                                   \
    } లేకపోతే {                                                                                                                      \
        OP_NAME##_uu(dst, src, val, అంకె_elems);                                                                                   \
    }                                                                                                                             \
}

HVX_ద్విమాన_SCALAR_DISPATCHER(hvx_కూడు_scalar_f32, దశాంశం)
HVX_ద్విమాన_SCALAR_DISPATCHER(hvx_sub_scalar_f32, దశాంశం)
HVX_ద్విమాన_SCALAR_DISPATCHER(hvx_mul_scalar_f32, దశాంశం)

HVX_ద్విమాన_SCALAR_DISPATCHER(hvx_కూడు_scalar_f16, _Float16)
HVX_ద్విమాన_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_ద్విమాన_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#నిర్వచించు HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, అంకె_elems);
    } లేకపోతే {
        hvx_min_scalar_f32_uu(dst, src, val, అంకె_elems);
    }
}

// MAX Scalar variants

#నిర్వచించు HVX_OP_MAX_SCALAR(v) Q6_Vsf_vmax_VsfVsf(val_vec, v)

స్థిర అంతఃవరుస శూన్య hvx_max_scalar_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_max_scalar_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_MAX_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_max_scalar_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_max_scalar_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_MAX_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_max_scalar_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం val, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {
        hvx_max_scalar_f32_aa(dst, src, val, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        hvx_max_scalar_f32_au(dst, src, val, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        hvx_max_scalar_f32_ua(dst, src, val, అంకె_elems);
    } లేకపోతే {
        hvx_max_scalar_f32_uu(dst, src, val, అంకె_elems);
    }
}

#నిర్వచించు HVX_OP_MIN_SCALAR_F16(v) Q6_Vhf_vmin_VhfVhf(val_vec, v)

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f16(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_సదిశ, పరిమాణం(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f16(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_UVector, పరిమాణం(_Float16), hvx_vec_store_a, HVX_OP_MIN_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f16(val);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_సదిశ, పరిమాణం(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec = hvx_vec_splat_f16(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, పరిమాణం(_Float16), hvx_vec_store_u, HVX_OP_MIN_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_min_scalar_f16(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {
        hvx_min_scalar_f16_aa(dst, src, val, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        hvx_min_scalar_f16_au(dst, src, val, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        hvx_min_scalar_f16_ua(dst, src, val, అంకె_elems);
    } లేకపోతే {
        hvx_min_scalar_f16_uu(dst, src, val, అంకె_elems);
    }
}

// CLAMP Scalar variants

#నిర్వచించు HVX_OP_CLAMP_SCALAR(v)                                           \
    ({                                                                   \
        HVX_సదిశ_Pred pred_cap_కుడి = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_సదిశ_Pred pred_cap_ఎడమ  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_సదిశ tmp = Q6_V_vmux_QVV(pred_cap_కుడి, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ఎడమ, min_vec, tmp);                      \
    })

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం min, స్థిరరాశి దశాంశం max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f32(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f32(max);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం min, స్థిరరాశి దశాంశం max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f32(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f32(max);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం min, స్థిరరాశి దశాంశం max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f32(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f32(max);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం min, స్థిరరాశి దశాంశం max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f32(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం min, స్థిరరాశి దశాంశం max, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, అంకె_elems);
    } లేకపోతే {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, అంకె_elems);
    }
}

#నిర్వచించు HVX_OP_CLAMP_SCALAR_F16(v)                                       \
    ({                                                                   \
        HVX_సదిశ_Pred pred_cap_కుడి = Q6_Q_vcmp_gt_VhfVhf(v, max_vec); \
        HVX_సదిశ_Pred pred_cap_ఎడమ  = Q6_Q_vcmp_gt_VhfVhf(min_vec, v); \
        HVX_సదిశ tmp = Q6_V_vmux_QVV(pred_cap_కుడి, max_vec, v);      \
        Q6_V_vmux_QVV(pred_cap_ఎడమ, min_vec, tmp);                      \
    })

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 min, స్థిరరాశి _Float16 max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f16(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f16(max);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_సదిశ, పరిమాణం(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 min, స్థిరరాశి _Float16 max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f16(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f16(max);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_UVector, పరిమాణం(_Float16), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 min, స్థిరరాశి _Float16 max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f16(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f16(max);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_సదిశ, పరిమాణం(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 min, స్థిరరాశి _Float16 max, uint32_t n) {
    స్థిరరాశి HVX_సదిశ min_vec = hvx_vec_splat_f16(min);
    స్థిరరాశి HVX_సదిశ max_vec = hvx_vec_splat_f16(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, పరిమాణం(_Float16), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR_F16);
}

స్థిర అంతఃవరుస శూన్య hvx_clamp_scalar_f16(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 min, స్థిరరాశి _Float16 max, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {
        hvx_clamp_scalar_f16_aa(dst, src, min, max, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        hvx_clamp_scalar_f16_au(dst, src, min, max, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        hvx_clamp_scalar_f16_ua(dst, src, min, max, అంకె_elems);
    } లేకపోతే {
        hvx_clamp_scalar_f16_uu(dst, src, min, max, అంకె_elems);
    }
}

#నిర్వచించు HVX_OP_LEAKY_RELU_SCALAR(v)                                 \
    ({                                                              \
        HVX_సదిశ_Pred pred_neg = Q6_Q_vcmp_gt_VsfVsf(zero_vec, v); \
        HVX_సదిశ     scaled   = HVX_OP_MUL_F32(v, ns_vec);        \
        Q6_V_vmux_QVV(pred_neg, scaled, v);                         \
    })

స్థిర అంతఃవరుస శూన్య hvx_leaky_relu_scalar_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం ns, uint32_t n) {
    స్థిరరాశి HVX_సదిశ zero_vec = hvx_vec_splat_f32(0.0f);
    స్థిరరాశి HVX_సదిశ ns_vec   = hvx_vec_splat_f32(ns);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_leaky_relu_scalar_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం ns, uint32_t n) {
    స్థిరరాశి HVX_సదిశ zero_vec = hvx_vec_splat_f32(0.0f);
    స్థిరరాశి HVX_సదిశ ns_vec   = hvx_vec_splat_f32(ns);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_సదిశ, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_a, HVX_OP_LEAKY_RELU_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_leaky_relu_scalar_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం ns, uint32_t n) {
    స్థిరరాశి HVX_సదిశ zero_vec = hvx_vec_splat_f32(0.0f);
    స్థిరరాశి HVX_సదిశ ns_vec   = hvx_vec_splat_f32(ns);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_సదిశ, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_leaky_relu_scalar_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం ns, uint32_t n) {
    స్థిరరాశి HVX_సదిశ zero_vec = hvx_vec_splat_f32(0.0f);
    స్థిరరాశి HVX_సదిశ ns_vec   = hvx_vec_splat_f32(ns);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, పరిమాణం(దశాంశం), hvx_vec_store_u, HVX_OP_LEAKY_RELU_SCALAR);
}

స్థిర అంతఃవరుస శూన్య hvx_leaky_relu_scalar_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి దశాంశం ns, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {
        hvx_leaky_relu_scalar_f32_aa(dst, src, ns, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        hvx_leaky_relu_scalar_f32_au(dst, src, ns, అంకె_elems);
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
        hvx_leaky_relu_scalar_f32_ua(dst, src, ns, అంకె_elems);
    } లేకపోతే {
        hvx_leaky_relu_scalar_f32_uu(dst, src, ns, అంకె_elems);
    }
}

//
// Abs
//

స్థిర అంతఃవరుస శూన్య hvx_abs_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t epv       = 128 / elem_కొలత;
    స్థిరరాశి uint32_t nvec      = n / epv;
    స్థిరరాశి uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        vdst[i] = hvx_vec_abs_f32(vsrc[i]);
    }
    ఒకవేళ (nloe) {
        HVX_సదిశ v = hvx_vec_abs_f32(vsrc[i]);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * elem_కొలత, v);
    }
}

#నిర్వచించు hvx_abs_f16_loop_body(dst_type, src_type, vec_store)               \
    చేయి {                                                                   \
        dst_type * పరిమిత vdst  = (dst_type *) dst;                      \
        src_type * పరిమిత vsrc = (src_type *) src;                       \
                                                                           \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(_Float16);                       \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                             \
        స్థిరరాశి uint32_t nvec = n / epv;                                     \
        స్థిరరాశి uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        కొరకు (; i < nvec; i++) {                                            \
            vdst[i] = hvx_vec_abs_f16(vsrc[i]);                            \
        }                                                                  \
        ఒకవేళ (nloe) {                                                        \
            HVX_సదిశ v = hvx_vec_abs_f16(vsrc[i]);                       \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, v);             \
        }                                                                  \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_abs_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_abs_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_abs_f16_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_abs_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_abs_f16_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_abs_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_abs_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_abs_f16(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి uint32_t అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
            hvx_abs_f16_aa(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_abs_f16_au(dst, src, అంకె_elems);
        }
    } లేకపోతే {
        ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
            hvx_abs_f16_ua(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_abs_f16_uu(dst, src, అంకె_elems);
        }
    }
}

//
//అడుగు
//

స్థిర అంతఃవరుస శూన్య hvx_అడుగు_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t epv       = 128 / elem_కొలత;
    స్థిరరాశి uint32_t nvec      = n / epv;
    స్థిరరాశి uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        vdst[i] = hvx_vec_అడుగు_f32(vsrc[i]);
    }
    ఒకవేళ (nloe) {
        HVX_సదిశ v = hvx_vec_అడుగు_f32(vsrc[i]);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * elem_కొలత, v);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_అడుగు_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(_Float16);
    స్థిరరాశి uint32_t epv       = 128 / elem_కొలత;
    స్థిరరాశి uint32_t nvec      = n / epv;
    స్థిరరాశి uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        vdst[i] = hvx_vec_అడుగు_f16(vsrc[i]);
    }
    ఒకవేళ (nloe) {
        HVX_సదిశ v = hvx_vec_అడుగు_f16(vsrc[i]);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * elem_కొలత, v);
    }
}

//
//వర్గం
//

#నిర్వచించు hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)               \
    చేయి {                                                                   \
        dst_type * పరిమిత vdst  = (dst_type *) dst;                      \
        src_type * పరిమిత vsrc = (src_type *) src;                       \
                                                                           \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);                          \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                             \
        స్థిరరాశి uint32_t nvec = n / epv;                                     \
        స్థిరరాశి uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        కొరకు (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ఒకవేళ (nloe) {                                                        \
            HVX_సదిశ v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);               \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, v);             \
        }                                                                  \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_sqr_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f32(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి uint32_t అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqr_f32_au(dst, src, అంకె_elems);
        }
    } లేకపోతే {
        ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqr_f32_uu(dst, src, అంకె_elems);
        }
    }
}

#నిర్వచించు hvx_sqr_f16_loop_body(dst_type, src_type, vec_store)               \
    చేయి {                                                                   \
        dst_type * పరిమిత vdst  = (dst_type *) dst;                      \
        src_type * పరిమిత vsrc = (src_type *) src;                       \
                                                                           \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(_Float16);                       \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                             \
        స్థిరరాశి uint32_t nvec = n / epv;                                     \
        స్థిరరాశి uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        కొరకు (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);                    \
        }                                                                  \
        ఒకవేళ (nloe) {                                                        \
            HVX_సదిశ v = HVX_OP_MUL_F16(vsrc[i], vsrc[i]);               \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, v);             \
        }                                                                  \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_sqr_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_sqr_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sqr_f16(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి uint32_t అంకె_elems) {
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {
        ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
            hvx_sqr_f16_aa(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqr_f16_au(dst, src, అంకె_elems);
        }
    } లేకపోతే {
        ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {
            hvx_sqr_f16_ua(dst, src, అంకె_elems);
        } లేకపోతే {
            hvx_sqr_f16_uu(dst, src, అంకె_elems);
        }
    }
}

#నిర్వచనం_రద్దు HVX_OP_కూడు_F32
#నిర్వచనం_రద్దు HVX_OP_SUB_F32
#నిర్వచనం_రద్దు HVX_OP_MUL_F32
#నిర్వచనం_రద్దు HVX_OP_కూడు_F16
#నిర్వచనం_రద్దు HVX_OP_SUB_F16
#నిర్వచనం_రద్దు HVX_OP_MUL_F16
#నిర్వచనం_రద్దు hvx_arith_loop_body
#నిర్వచనం_రద్దు HVX_OP_కూడు_SCALAR_F32
#నిర్వచనం_రద్దు HVX_OP_SUB_SCALAR_F32
#నిర్వచనం_రద్దు HVX_OP_MUL_SCALAR_F32
#నిర్వచనం_రద్దు HVX_OP_కూడు_SCALAR_F16
#నిర్వచనం_రద్దు HVX_OP_SUB_SCALAR_F16
#నిర్వచనం_రద్దు HVX_OP_MUL_SCALAR_F16
#నిర్వచనం_రద్దు hvx_scalar_loop_body
#నిర్వచనం_రద్దు HVX_OP_MIN_SCALAR
#నిర్వచనం_రద్దు HVX_OP_MAX_SCALAR
#నిర్వచనం_రద్దు HVX_OP_MIN_SCALAR_F16
#నిర్వచనం_రద్దు HVX_OP_CLAMP_SCALAR
#నిర్వచనం_రద్దు HVX_OP_CLAMP_SCALAR_F16
#నిర్వచనం_రద్దు HVX_OP_LEAKY_RELU_SCALAR
#నిర్వచనం_రద్దు DEFINE_HVX_ద్విమాన_OP_VARIANTS
#నిర్వచనం_రద్దు DEFINE_HVX_ద్విమాన_SCALAR_OP_VARIANTS
#నిర్వచనం_రద్దు HVX_ద్విమాన_DISPATCHER
#నిర్వచనం_రద్దు వాడని

#అయితే_ముగింపు // HVX_ARITH_H
