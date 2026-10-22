#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_DIV_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_DIV_H

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-inverse.h"
#ಅನ್ತರ್ಭಾವ "hvx-arith.h"

#ಆದರೆ __HVX_ARCH__ < 79
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#ಆದರೆ_ಅಂತ್ಯ

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_div_mul_f16_const_using_f32(HVX_ಸದಿಶ vec1_hf, HVX_ಸದಿಶ vec2_sf_const, HVX_ಸದಿಶ vec_hf_one_1_0) {
#ಆದರೆ __HVX_ARCH__ < 79
    HVX_ಸದಿಶ_Pair src_ಗಮ್ಯ_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ಸದಿಶ src_ಗಮ್ಯ_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_ಗಮ್ಯ_f32));
    HVX_ಸದಿಶ src_ಗಮ್ಯ_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_ಗಮ್ಯ_f32));
#ಇಲ್ಲದಿದ್ದರೆ
    HVX_ಸದಿಶ_Pair src_ಗಮ್ಯ_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ಸದಿಶ src_ಗಮ್ಯ_f32_0 = Q6_V_lo_W(src_ಗಮ್ಯ_f32);
    HVX_ಸದಿಶ src_ಗಮ್ಯ_f32_1 = Q6_V_hi_W(src_ಗಮ್ಯ_f32);
#ಆದರೆ_ಅಂತ್ಯ

    HVX_ಸದಿಶ div_f32_0 = HVX_OP_MUL_F32(src_ಗಮ್ಯ_f32_0, vec2_sf_const);
    HVX_ಸದಿಶ div_f32_1 = HVX_OP_MUL_F32(src_ಗಮ್ಯ_f32_1, vec2_sf_const);

#ಆದರೆ __HVX_ARCH__ < 79
    HVX_ಸದಿಶ res = hvx_vec_f32_ಗಮ್ಯ_f16(div_f32_0, div_f32_1);
#ಇಲ್ಲದಿದ್ದರೆ
    HVX_ಸದಿಶ res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_div_mul_f16_const_using_f16(HVX_ಸದಿಶ vec1_hf, HVX_ಸದಿಶ const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    ಹಿಂತಿರುಗಿ HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    ಮಾಡು {                                                                                               \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                                   \
        src_type * ಸೀಮಿತ vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_ಸದಿಶ hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP16;                                                           \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        ಗಾಗಿ (; i < nvec; i++) {                                                                        \
            HVX_ಸದಿಶ res;                                                                            \
            ಆದರೆ (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } ಇಲ್ಲದಿದ್ದರೆ {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        ಆದರೆ (nloe) {                                                                                    \
            HVX_ಸದಿಶ res;                                                                            \
            ಆದರೆ (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } ಇಲ್ಲದಿದ್ದರೆ {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } ಆಗ(0)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_div_scalar_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ದಶಮಾಂಶ)val));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_div_scalar_f16_au(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ದಶಮಾಂಶ)val));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_div_scalar_f16_ua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ದಶಮಾಂಶ)val));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_div_scalar_f16_uu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f32 = hvx_vec_splat_f32(1.0f/((ದಶಮಾಂಶ)val));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_div_f16_using_f32(HVX_ಸದಿಶ vec1, HVX_ಸದಿಶ vec2, HVX_ಸದಿಶ f32_nan_inf_ಮುಖವಾಡ, HVX_ಸದಿಶ vec_hf_one_1_0) {
#ಆದರೆ __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_ಸದಿಶ_Pair vec1_ಗಮ್ಯ_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ಸದಿಶ     vec1_ಗಮ್ಯ_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_ಗಮ್ಯ_f32));
    HVX_ಸದಿಶ     vec1_ಗಮ್ಯ_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_ಗಮ್ಯ_f32));

    // Convert second input to fp32
    HVX_ಸದಿಶ_Pair vec2_ಗಮ್ಯ_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ಸದಿಶ     vec2_ಗಮ್ಯ_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_ಗಮ್ಯ_f32));
    HVX_ಸದಿಶ     vec2_ಗಮ್ಯ_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_ಗಮ್ಯ_f32));
#ಇಲ್ಲದಿದ್ದರೆ
    // Convert first input to fp32
    HVX_ಸದಿಶ_Pair vec1_ಗಮ್ಯ_f32   = Q6_Wsf_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ಸದಿಶ     vec1_ಗಮ್ಯ_f32_0 = Q6_V_lo_W(vec1_ಗಮ್ಯ_f32);
    HVX_ಸದಿಶ     vec1_ಗಮ್ಯ_f32_1 = Q6_V_hi_W(vec1_ಗಮ್ಯ_f32);

    // Convert second input to fp32
    HVX_ಸದಿಶ_Pair vec2_ಗಮ್ಯ_f32   = Q6_Wsf_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ಸದಿಶ     vec2_ಗಮ್ಯ_f32_0 = Q6_V_lo_W(vec2_ಗಮ್ಯ_f32);
    HVX_ಸದಿಶ     vec2_ಗಮ್ಯ_f32_1 = Q6_V_hi_W(vec2_ಗಮ್ಯ_f32);
#ಆದರೆ_ಅಂತ್ಯ

    // Inverse second input in fp32
    HVX_ಸದಿಶ     vec2_inv_f32_0 = hvx_vec_inverse_f32_guard(vec2_ಗಮ್ಯ_f32_0, f32_nan_inf_ಮುಖವಾಡ);
    HVX_ಸದಿಶ     vec2_inv_f32_1 = hvx_vec_inverse_f32_guard(vec2_ಗಮ್ಯ_f32_1, f32_nan_inf_ಮುಖವಾಡ);

    // Multiply first input by inverse of second, in fp32
    HVX_ಸದಿಶ     div_f32_0 = HVX_OP_MUL_F32(vec1_ಗಮ್ಯ_f32_0, vec2_inv_f32_0);
    HVX_ಸದಿಶ     div_f32_1 = HVX_OP_MUL_F32(vec1_ಗಮ್ಯ_f32_1, vec2_inv_f32_1);

    // Convert back to fp16
#ಆದರೆ __HVX_ARCH__ < 79
    HVX_ಸದಿಶ     recip = hvx_vec_f32_ಗಮ್ಯ_f16(div_f32_0, div_f32_1);
#ಇಲ್ಲದಿದ್ದರೆ
    HVX_ಸದಿಶ     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_hybrid_div_f16(HVX_ಸದಿಶ vec1, HVX_ಸದಿಶ vec2, HVX_ಸದಿಶ f32_nan_inf_ಮುಖವಾಡ, HVX_ಸದಿಶ f16_nan_inf_ಮುಖವಾಡ, HVX_ಸದಿಶ vec_hf_one_1_0) {
#ಆದರೆ __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_ಸದಿಶ vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_ಮುಖವಾಡ);
    ಹಿಂತಿರುಗಿ HVX_OP_MUL_F16(vec1, vec2_inv);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_ಮುಖವಾಡ, vec_hf_one_1_0);
#ಆದರೆ_ಅಂತ್ಯ
}

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    ಮಾಡು {                                                                                  \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                      \
        src0_type * ಸೀಮಿತ vsrc0 = (src0_type *) src0;                                  \
        src1_type * ಸೀಮಿತ vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ f32_nan_inf_ಮುಖವಾಡ = Q6_V_vsplat_R(0x7f800000);                    \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ f16_nan_inf_ಮುಖವಾಡ = Q6_Vh_vsplat_R(0x7c00);                       \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP16;                                              \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ಗಾಗಿ (; i < nvec; i++) {                                                           \
            HVX_ಸದಿಶ res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ಮುಖವಾಡ, f16_nan_inf_ಮುಖವಾಡ,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        ಆದರೆ (nloe) {                                                                       \
            HVX_ಸದಿಶ res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ಮುಖವಾಡ, f16_nan_inf_ಮುಖವಾಡ,   \
                                                    hf_one);                              \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } ಆಗ(0)

#ವ್ಯಾಖ್ಯಾನಿಸು hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    ಮಾಡು {                                                                             \
        dst_type * ಸೀಮಿತ vdst = (dst_type *) dst;                                 \
        src0_type * ಸೀಮಿತ vsrc0 = (src0_type *) src0;                             \
        src1_type * ಸೀಮಿತ vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ nan_inf_ಮುಖವಾಡ = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP32;                                         \
        ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        ಗಾಗಿ (; i < nvec; i++) {                                                      \
            HVX_ಸದಿಶ inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ಮುಖವಾಡ); \
            HVX_ಸದಿಶ res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        ಆದರೆ (nloe) {                                                                  \
            HVX_ಸದಿಶ inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ಮುಖವಾಡ); \
            HVX_ಸದಿಶ res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((ಶೂನ್ಯ *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } ಆಗ(0)

// Generic macro to define alignment permutations for an op
#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_aaa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                              \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ಸದಿಶ, HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_aau(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                              \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ಸದಿಶ, HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_aua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                              \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ಸದಿಶ, HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_auu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_ಸದಿಶ, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uaa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ಸದಿಶ, HVX_ಸದಿಶ, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uau(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ಸದಿಶ, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uua(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    ಪ್ರತಿಜ್ಞಾ((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_ಸದಿಶ, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME##_uuu(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ OP_NAME(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) { \
    ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) dst, 128)) {                                                                                                 \
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src0, 128)) {                                                                                            \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_aau(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                                             \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_auu(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        }                                                                                                                                    \
    } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                                                 \
        ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src0, 128)) {                                                                                            \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_uau(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                                             \
            ಆದರೆ (hex_is_aligned((ಶೂನ್ಯ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, ಅಂಕೆ_elems);                                               \
            ಇಲ್ಲದಿದ್ದರೆ                                    OP_NAME##_uuu(dst, src0, src1, ಅಂಕೆ_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MUL_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MUL_F16

#ಆದರೆ_ಅಂತ್ಯ // HVX_DIV_H
