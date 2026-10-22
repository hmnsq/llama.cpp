#నిర్వచితం_కాకపోతే HVX_DIV_H
#నిర్వచించు HVX_DIV_H

#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "hvx-inverse.h"
#అన్తర్భావ "hvx-arith.h"

#ఒకవేళ __HVX_ARCH__ < 79
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#నిర్వచించు HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#లేకపోతే
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#నిర్వచించు HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#అయితే_ముగింపు

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
స్థిర అంతఃవరుస HVX_సదిశ hvx_div_mul_f16_const_using_f32(HVX_సదిశ vec1_hf, HVX_సదిశ vec2_sf_const, HVX_సదిశ vec_hf_one_1_0) {
#ఒకవేళ __HVX_ARCH__ < 79
    HVX_సదిశ_Pair src_గమ్యం_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_సదిశ src_గమ్యం_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_గమ్యం_f32));
    HVX_సదిశ src_గమ్యం_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_గమ్యం_f32));
#లేకపోతే
    HVX_సదిశ_Pair src_గమ్యం_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_సదిశ src_గమ్యం_f32_0 = Q6_V_lo_W(src_గమ్యం_f32);
    HVX_సదిశ src_గమ్యం_f32_1 = Q6_V_hi_W(src_గమ్యం_f32);
#అయితే_ముగింపు

    HVX_సదిశ div_f32_0 = HVX_OP_MUL_F32(src_గమ్యం_f32_0, vec2_sf_const);
    HVX_సదిశ div_f32_1 = HVX_OP_MUL_F32(src_గమ్యం_f32_1, vec2_sf_const);

#ఒకవేళ __HVX_ARCH__ < 79
    HVX_సదిశ res = hvx_vec_f32_గమ్యం_f16(div_f32_0, div_f32_1);
#లేకపోతే
    HVX_సదిశ res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#అయితే_ముగింపు
    తిరిగి res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
స్థిర అంతఃవరుస HVX_సదిశ hvx_div_mul_f16_const_using_f16(HVX_సదిశ vec1_hf, HVX_సదిశ const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    తిరిగి HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#నిర్వచించు hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    చేయి {                                                                                               \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                                   \
        src_type * పరిమిత vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_సదిశ hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP16;                                                           \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        కొరకు (; i < nvec; i++) {                                                                        \
            HVX_సదిశ res;                                                                            \
            ఒకవేళ (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } లేకపోతే {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        ఒకవేళ (nloe) {                                                                                    \
            HVX_సదిశ res;                                                                            \
            ఒకవేళ (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } లేకపోతే {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_div_scalar_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec_f32 = hvx_vec_splat_f32(1.0f/((దశాంశం)val));
    స్థిరరాశి HVX_సదిశ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}
స్థిర అంతఃవరుస శూన్య hvx_div_scalar_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec_f32 = hvx_vec_splat_f32(1.0f/((దశాంశం)val));
    స్థిరరాశి HVX_సదిశ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}
స్థిర అంతఃవరుస శూన్య hvx_div_scalar_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec_f32 = hvx_vec_splat_f32(1.0f/((దశాంశం)val));
    స్థిరరాశి HVX_సదిశ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}
స్థిర అంతఃవరుస శూన్య hvx_div_scalar_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి _Float16 val, uint32_t n) {
    స్థిరరాశి HVX_సదిశ val_vec_f32 = hvx_vec_splat_f32(1.0f/((దశాంశం)val));
    స్థిరరాశి HVX_సదిశ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_div_f16_using_f32(HVX_సదిశ vec1, HVX_సదిశ vec2, HVX_సదిశ f32_nan_inf_ముసుగు, HVX_సదిశ vec_hf_one_1_0) {
#ఒకవేళ __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_సదిశ_Pair vec1_గమ్యం_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_సదిశ     vec1_గమ్యం_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_గమ్యం_f32));
    HVX_సదిశ     vec1_గమ్యం_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_గమ్యం_f32));

    // Convert second input to fp32
    HVX_సదిశ_Pair vec2_గమ్యం_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_సదిశ     vec2_గమ్యం_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_గమ్యం_f32));
    HVX_సదిశ     vec2_గమ్యం_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_గమ్యం_f32));
#లేకపోతే
    // Convert first input to fp32
    HVX_సదిశ_Pair vec1_గమ్యం_f32   = Q6_Wsf_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_సదిశ     vec1_గమ్యం_f32_0 = Q6_V_lo_W(vec1_గమ్యం_f32);
    HVX_సదిశ     vec1_గమ్యం_f32_1 = Q6_V_hi_W(vec1_గమ్యం_f32);

    // Convert second input to fp32
    HVX_సదిశ_Pair vec2_గమ్యం_f32   = Q6_Wsf_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_సదిశ     vec2_గమ్యం_f32_0 = Q6_V_lo_W(vec2_గమ్యం_f32);
    HVX_సదిశ     vec2_గమ్యం_f32_1 = Q6_V_hi_W(vec2_గమ్యం_f32);
#అయితే_ముగింపు

    // Inverse second input in fp32
    HVX_సదిశ     vec2_inv_f32_0 = hvx_vec_inverse_f32_guard(vec2_గమ్యం_f32_0, f32_nan_inf_ముసుగు);
    HVX_సదిశ     vec2_inv_f32_1 = hvx_vec_inverse_f32_guard(vec2_గమ్యం_f32_1, f32_nan_inf_ముసుగు);

    // Multiply first input by inverse of second, in fp32
    HVX_సదిశ     div_f32_0 = HVX_OP_MUL_F32(vec1_గమ్యం_f32_0, vec2_inv_f32_0);
    HVX_సదిశ     div_f32_1 = HVX_OP_MUL_F32(vec1_గమ్యం_f32_1, vec2_inv_f32_1);

    // Convert back to fp16
#ఒకవేళ __HVX_ARCH__ < 79
    HVX_సదిశ     recip = hvx_vec_f32_గమ్యం_f16(div_f32_0, div_f32_1);
#లేకపోతే
    HVX_సదిశ     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#అయితే_ముగింపు

    తిరిగి recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_hybrid_div_f16(HVX_సదిశ vec1, HVX_సదిశ vec2, HVX_సదిశ f32_nan_inf_ముసుగు, HVX_సదిశ f16_nan_inf_ముసుగు, HVX_సదిశ vec_hf_one_1_0) {
#ఒకవేళ __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_సదిశ vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_ముసుగు);
    తిరిగి HVX_OP_MUL_F16(vec1, vec2_inv);
#లేకపోతే
    తిరిగి hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_ముసుగు, vec_hf_one_1_0);
#అయితే_ముగింపు
}

#నిర్వచించు hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    చేయి {                                                                                  \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                      \
        src0_type * పరిమిత vsrc0 = (src0_type *) src0;                                  \
        src1_type * పరిమిత vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        స్థిరరాశి HVX_సదిశ f32_nan_inf_ముసుగు = Q6_V_vsplat_R(0x7f800000);                    \
        స్థిరరాశి HVX_సదిశ f16_nan_inf_ముసుగు = Q6_Vh_vsplat_R(0x7c00);                       \
        స్థిరరాశి HVX_సదిశ hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP16;                                              \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        కొరకు (; i < nvec; i++) {                                                           \
            HVX_సదిశ res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ముసుగు, f16_nan_inf_ముసుగు,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        ఒకవేళ (nloe) {                                                                       \
            HVX_సదిశ res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ముసుగు, f16_nan_inf_ముసుగు,   \
                                                    hf_one);                              \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } అయితే(0)

#నిర్వచించు hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    చేయి {                                                                             \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                 \
        src0_type * పరిమిత vsrc0 = (src0_type *) src0;                             \
        src1_type * పరిమిత vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        స్థిరరాశి HVX_సదిశ nan_inf_ముసుగు = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP32;                                         \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        కొరకు (; i < nvec; i++) {                                                      \
            HVX_సదిశ inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ముసుగు); \
            HVX_సదిశ res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        ఒకవేళ (nloe) {                                                                  \
            HVX_సదిశ inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ముసుగు); \
            HVX_సదిశ res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } అయితే(0)

// Generic macro to define alignment permutations for an op
#నిర్వచించు DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aaa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                                              \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                                             \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_సదిశ, HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aau(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                                              \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_సదిశ, HVX_సదిశ, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                                              \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_సదిశ, HVX_UVector, HVX_సదిశ, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
స్థిర అంతఃవరుస శూన్య OP_NAME##_auu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_సదిశ, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uaa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                                             \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_సదిశ, HVX_సదిశ, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uau(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_సదిశ, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_సదిశ, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uuu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#నిర్వచించు HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
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

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#నిర్వచనం_రద్దు HVX_OP_MUL_F32
#నిర్వచనం_రద్దు HVX_OP_MUL_F16

#అయితే_ముగింపు // HVX_DIV_H
