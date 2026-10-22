#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_DIV_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_DIV_H

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-inverse.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-arith.h"

#ਜੇ __HVX_ARCH__ < 79
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#ਜੇ_ਅੰਤ

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_div_mul_f16_const_using_f32(HVX_ਸਦਿਸ਼ vec1_hf, HVX_ਸਦਿਸ਼ vec2_sf_const, HVX_ਸਦਿਸ਼ vec_hf_one_1_0) {
#ਜੇ __HVX_ARCH__ < 79
    HVX_ਸਦਿਸ਼_Pair src_ਮੰਜ਼ਿਲ_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ਸਦਿਸ਼ src_ਮੰਜ਼ਿਲ_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_ਮੰਜ਼ਿਲ_f32));
    HVX_ਸਦਿਸ਼ src_ਮੰਜ਼ਿਲ_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_ਮੰਜ਼ਿਲ_f32));
#ਨਹੀਂਤਾਂ
    HVX_ਸਦਿਸ਼_Pair src_ਮੰਜ਼ਿਲ_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ਸਦਿਸ਼ src_ਮੰਜ਼ਿਲ_f32_0 = Q6_V_lo_W(src_ਮੰਜ਼ਿਲ_f32);
    HVX_ਸਦਿਸ਼ src_ਮੰਜ਼ਿਲ_f32_1 = Q6_V_hi_W(src_ਮੰਜ਼ਿਲ_f32);
#ਜੇ_ਅੰਤ

    HVX_ਸਦਿਸ਼ div_f32_0 = HVX_OP_MUL_F32(src_ਮੰਜ਼ਿਲ_f32_0, vec2_sf_const);
    HVX_ਸਦਿਸ਼ div_f32_1 = HVX_OP_MUL_F32(src_ਮੰਜ਼ਿਲ_f32_1, vec2_sf_const);

#ਜੇ __HVX_ARCH__ < 79
    HVX_ਸਦਿਸ਼ res = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(div_f32_0, div_f32_1);
#ਨਹੀਂਤਾਂ
    HVX_ਸਦਿਸ਼ res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#ਜੇ_ਅੰਤ
    ਵਾਪਸ res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_div_mul_f16_const_using_f16(HVX_ਸਦਿਸ਼ vec1_hf, HVX_ਸਦਿਸ਼ const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    ਵਾਪਸ HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    ਕਰੋ {                                                                                               \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                                   \
        src_type * ਸੀਮਿਤ vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_ਸਦਿਸ਼ hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        ਅਚਲ uint32_t nvec = n / VLEN_FP16;                                                           \
        ਅਚਲ uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        ਲਈ (; i < nvec; i++) {                                                                        \
            HVX_ਸਦਿਸ਼ res;                                                                            \
            ਜੇ (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } ਨਹੀਂਤਾਂ {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        ਜੇ (nloe) {                                                                                    \
            HVX_ਸਦਿਸ਼ res;                                                                            \
            ਜੇ (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } ਨਹੀਂਤਾਂ {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } ਜਦੋਂ(0)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_div_scalar_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ਦਸ਼ਮਲਵ)val));
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);
}
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_div_scalar_f16_au(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ਦਸ਼ਮਲਵ)val));
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);
}
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_div_scalar_f16_ua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ਦਸ਼ਮਲਵ)val));
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);
}
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_div_scalar_f16_uu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ਦਸ਼ਮਲਵ)val));
    ਅਚਲ HVX_ਸਦਿਸ਼ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_div_f16_using_f32(HVX_ਸਦਿਸ਼ vec1, HVX_ਸਦਿਸ਼ vec2, HVX_ਸਦਿਸ਼ f32_nan_inf_ਨਕਾਬ, HVX_ਸਦਿਸ਼ vec_hf_one_1_0) {
#ਜੇ __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_ਸਦਿਸ਼_Pair vec1_ਮੰਜ਼ਿਲ_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ਸਦਿਸ਼     vec1_ਮੰਜ਼ਿਲ_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_ਮੰਜ਼ਿਲ_f32));
    HVX_ਸਦਿਸ਼     vec1_ਮੰਜ਼ਿਲ_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_ਮੰਜ਼ਿਲ_f32));

    // Convert second input to fp32
    HVX_ਸਦਿਸ਼_Pair vec2_ਮੰਜ਼ਿਲ_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ਸਦਿਸ਼     vec2_ਮੰਜ਼ਿਲ_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_ਮੰਜ਼ਿਲ_f32));
    HVX_ਸਦਿਸ਼     vec2_ਮੰਜ਼ਿਲ_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_ਮੰਜ਼ਿਲ_f32));
#ਨਹੀਂਤਾਂ
    // Convert first input to fp32
    HVX_ਸਦਿਸ਼_Pair vec1_ਮੰਜ਼ਿਲ_f32   = Q6_Wsf_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ਸਦਿਸ਼     vec1_ਮੰਜ਼ਿਲ_f32_0 = Q6_V_lo_W(vec1_ਮੰਜ਼ਿਲ_f32);
    HVX_ਸਦਿਸ਼     vec1_ਮੰਜ਼ਿਲ_f32_1 = Q6_V_hi_W(vec1_ਮੰਜ਼ਿਲ_f32);

    // Convert second input to fp32
    HVX_ਸਦਿਸ਼_Pair vec2_ਮੰਜ਼ਿਲ_f32   = Q6_Wsf_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ਸਦਿਸ਼     vec2_ਮੰਜ਼ਿਲ_f32_0 = Q6_V_lo_W(vec2_ਮੰਜ਼ਿਲ_f32);
    HVX_ਸਦਿਸ਼     vec2_ਮੰਜ਼ਿਲ_f32_1 = Q6_V_hi_W(vec2_ਮੰਜ਼ਿਲ_f32);
#ਜੇ_ਅੰਤ

    // Inverse second input in fp32
    HVX_ਸਦਿਸ਼     vec2_inv_f32_0 = hvx_vec_inverse_f32_guard(vec2_ਮੰਜ਼ਿਲ_f32_0, f32_nan_inf_ਨਕਾਬ);
    HVX_ਸਦਿਸ਼     vec2_inv_f32_1 = hvx_vec_inverse_f32_guard(vec2_ਮੰਜ਼ਿਲ_f32_1, f32_nan_inf_ਨਕਾਬ);

    // Multiply first input by inverse of second, in fp32
    HVX_ਸਦਿਸ਼     div_f32_0 = HVX_OP_MUL_F32(vec1_ਮੰਜ਼ਿਲ_f32_0, vec2_inv_f32_0);
    HVX_ਸਦਿਸ਼     div_f32_1 = HVX_OP_MUL_F32(vec1_ਮੰਜ਼ਿਲ_f32_1, vec2_inv_f32_1);

    // Convert back to fp16
#ਜੇ __HVX_ARCH__ < 79
    HVX_ਸਦਿਸ਼     recip = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(div_f32_0, div_f32_1);
#ਨਹੀਂਤਾਂ
    HVX_ਸਦਿਸ਼     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#ਜੇ_ਅੰਤ

    ਵਾਪਸ recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_hybrid_div_f16(HVX_ਸਦਿਸ਼ vec1, HVX_ਸਦਿਸ਼ vec2, HVX_ਸਦਿਸ਼ f32_nan_inf_ਨਕਾਬ, HVX_ਸਦਿਸ਼ f16_nan_inf_ਨਕਾਬ, HVX_ਸਦਿਸ਼ vec_hf_one_1_0) {
#ਜੇ __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_ਸਦਿਸ਼ vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_ਨਕਾਬ);
    ਵਾਪਸ HVX_OP_MUL_F16(vec1, vec2_inv);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_ਨਕਾਬ, vec_hf_one_1_0);
#ਜੇ_ਅੰਤ
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    ਕਰੋ {                                                                                  \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                      \
        src0_type * ਸੀਮਿਤ vsrc0 = (src0_type *) src0;                                  \
        src1_type * ਸੀਮਿਤ vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        ਅਚਲ HVX_ਸਦਿਸ਼ f32_nan_inf_ਨਕਾਬ = Q6_V_vsplat_R(0x7f800000);                    \
        ਅਚਲ HVX_ਸਦਿਸ਼ f16_nan_inf_ਨਕਾਬ = Q6_Vh_vsplat_R(0x7c00);                       \
        ਅਚਲ HVX_ਸਦਿਸ਼ hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        ਅਚਲ uint32_t nvec = n / VLEN_FP16;                                              \
        ਅਚਲ uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ਲਈ (; i < nvec; i++) {                                                           \
            HVX_ਸਦਿਸ਼ res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ਨਕਾਬ, f16_nan_inf_ਨਕਾਬ,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        ਜੇ (nloe) {                                                                       \
            HVX_ਸਦਿਸ਼ res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ਨਕਾਬ, f16_nan_inf_ਨਕਾਬ,   \
                                                    hf_one);                              \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } ਜਦੋਂ(0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    ਕਰੋ {                                                                             \
        dst_type * ਸੀਮਿਤ vdst = (dst_type *) dst;                                 \
        src0_type * ਸੀਮਿਤ vsrc0 = (src0_type *) src0;                             \
        src1_type * ਸੀਮਿਤ vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        ਅਚਲ HVX_ਸਦਿਸ਼ nan_inf_ਨਕਾਬ = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        ਅਚਲ uint32_t nvec = n / VLEN_FP32;                                         \
        ਅਚਲ uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        ਲਈ (; i < nvec; i++) {                                                      \
            HVX_ਸਦਿਸ਼ inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ਨਕਾਬ); \
            HVX_ਸਦਿਸ਼ res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        ਜੇ (nloe) {                                                                  \
            HVX_ਸਦਿਸ਼ inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ਨਕਾਬ); \
            HVX_ਸਦਿਸ਼ res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((ਖਾਲੀ *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } ਜਦੋਂ(0)

// Generic macro to define alignment permutations for an op
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_aaa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                                              \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_aau(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                                              \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_aua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                                              \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ਸਦਿਸ਼, HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_auu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_ਸਦਿਸ਼, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uaa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ਸਦਿਸ਼, HVX_ਸਦਿਸ਼, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uau(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ਸਦਿਸ਼, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uua(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    ਪ੍ਰਤਿਜ੍ਞਾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_ਸਦਿਸ਼, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME##_uuu(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ OP_NAME(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, ਅਚਲ uint32_t ਅੰਕ_elems) { \
    ਜੇ (hex_is_aligned((ਖਾਲੀ *) dst, 128)) {                                                                                                 \
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src0, 128)) {                                                                                            \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_aau(dst, src0, src1, ਅੰਕ_elems);                                               \
        } ਨਹੀਂਤਾਂ {                                                                                                                             \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_auu(dst, src0, src1, ਅੰਕ_elems);                                               \
        }                                                                                                                                    \
    } ਨਹੀਂਤਾਂ {                                                                                                                                 \
        ਜੇ (hex_is_aligned((ਖਾਲੀ *) src0, 128)) {                                                                                            \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_uau(dst, src0, src1, ਅੰਕ_elems);                                               \
        } ਨਹੀਂਤਾਂ {                                                                                                                             \
            ਜੇ (hex_is_aligned((ਖਾਲੀ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, ਅੰਕ_elems);                                               \
            ਨਹੀਂਤਾਂ                                    OP_NAME##_uuu(dst, src0, src1, ਅੰਕ_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MUL_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MUL_F16

#ਜੇ_ਅੰਤ // HVX_DIV_H
