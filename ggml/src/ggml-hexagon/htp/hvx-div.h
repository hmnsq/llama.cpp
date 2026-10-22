#ifndef HVX_DIV_H
#define HVX_DIV_H

#அந்தர்பாவ <HAP_farf.h>

#அந்தர்பாவ <math.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdint.h>

#அந்தர்பாவ "hvx-base.h"
#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "hvx-inverse.h"
#அந்தர்பாவ "hvx-arith.h"

#எனில் __HVX_ARCH__ < 79
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#இல்லையெனில்
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#endif

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
நிலையான உள்வரி HVX_Vector hvx_div_mul_f16_const_using_f32(HVX_Vector vec1_hf, HVX_Vector vec2_sf_const, HVX_Vector vec_hf_one_1_0) {
#எனில் __HVX_ARCH__ < 79
    HVX_VectorPair src_to_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_Vector src_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_to_f32));
    HVX_Vector src_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_to_f32));
#இல்லையெனில்
    HVX_VectorPair src_to_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_Vector src_to_f32_0 = Q6_V_lo_W(src_to_f32);
    HVX_Vector src_to_f32_1 = Q6_V_hi_W(src_to_f32);
#endif

    HVX_Vector div_f32_0 = HVX_OP_MUL_F32(src_to_f32_0, vec2_sf_const);
    HVX_Vector div_f32_1 = HVX_OP_MUL_F32(src_to_f32_1, vec2_sf_const);

#எனில் __HVX_ARCH__ < 79
    HVX_Vector res = hvx_vec_f32_to_f16(div_f32_0, div_f32_1);
#இல்லையெனில்
    HVX_Vector res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#endif
    திரும்பு res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
நிலையான உள்வரி HVX_Vector hvx_div_mul_f16_const_using_f16(HVX_Vector vec1_hf, HVX_Vector const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    திரும்பு HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#define hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    செய் {                                                                                               \
        dst_type * restrict vdst = (dst_type *) dst;                                                   \
        src_type * restrict vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        மாறிலி uint32_t nvec = n / VLEN_FP16;                                                           \
        மாறிலி uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        க்கு (; i < nvec; i++) {                                                                        \
            HVX_Vector res;                                                                            \
            எனில் (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } இல்லையெனில் {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        எனில் (nloe) {                                                                                    \
            HVX_Vector res;                                                                            \
            எனில் (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } இல்லையெனில் {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((வெறுமை *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } போது(0)

நிலையான உள்வரி வெறுமை hvx_div_scalar_f16_aa(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி _Float16 val, uint32_t n) {
    மாறிலி HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((தசம)val));
    மாறிலி HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ப்ரதிஜ்ஞா((uintptr_t) dst % 128 == 0);
    ப்ரதிஜ்ஞா((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}
நிலையான உள்வரி வெறுமை hvx_div_scalar_f16_au(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி _Float16 val, uint32_t n) {
    மாறிலி HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((தசம)val));
    மாறிலி HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ப்ரதிஜ்ஞா((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}
நிலையான உள்வரி வெறுமை hvx_div_scalar_f16_ua(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி _Float16 val, uint32_t n) {
    மாறிலி HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((தசம)val));
    மாறிலி HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ப்ரதிஜ்ஞா((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}
நிலையான உள்வரி வெறுமை hvx_div_scalar_f16_uu(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி _Float16 val, uint32_t n) {
    மாறிலி HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((தசம)val));
    மாறிலி HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
நிலையான உள்வரி HVX_Vector hvx_vec_div_f16_using_f32(HVX_Vector vec1, HVX_Vector vec2, HVX_Vector f32_nan_inf_mask, HVX_Vector vec_hf_one_1_0) {
#எனில் __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_VectorPair vec1_to_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec1_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_to_f32));
    HVX_Vector     vec1_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_to_f32));

    // Convert second input to fp32
    HVX_VectorPair vec2_to_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec2_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_to_f32));
    HVX_Vector     vec2_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_to_f32));
#இல்லையெனில்
    // Convert first input to fp32
    HVX_VectorPair vec1_to_f32   = Q6_Wsf_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec1_to_f32_0 = Q6_V_lo_W(vec1_to_f32);
    HVX_Vector     vec1_to_f32_1 = Q6_V_hi_W(vec1_to_f32);

    // Convert second input to fp32
    HVX_VectorPair vec2_to_f32   = Q6_Wsf_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec2_to_f32_0 = Q6_V_lo_W(vec2_to_f32);
    HVX_Vector     vec2_to_f32_1 = Q6_V_hi_W(vec2_to_f32);
#endif

    // Inverse second input in fp32
    HVX_Vector     vec2_inv_f32_0 = hvx_vec_inverse_f32_guard(vec2_to_f32_0, f32_nan_inf_mask);
    HVX_Vector     vec2_inv_f32_1 = hvx_vec_inverse_f32_guard(vec2_to_f32_1, f32_nan_inf_mask);

    // Multiply first input by inverse of second, in fp32
    HVX_Vector     div_f32_0 = HVX_OP_MUL_F32(vec1_to_f32_0, vec2_inv_f32_0);
    HVX_Vector     div_f32_1 = HVX_OP_MUL_F32(vec1_to_f32_1, vec2_inv_f32_1);

    // Convert back to fp16
#எனில் __HVX_ARCH__ < 79
    HVX_Vector     recip = hvx_vec_f32_to_f16(div_f32_0, div_f32_1);
#இல்லையெனில்
    HVX_Vector     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#endif

    திரும்பு recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
நிலையான உள்வரி HVX_Vector hvx_vec_hybrid_div_f16(HVX_Vector vec1, HVX_Vector vec2, HVX_Vector f32_nan_inf_mask, HVX_Vector f16_nan_inf_mask, HVX_Vector vec_hf_one_1_0) {
#எனில் __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_Vector vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_mask);
    திரும்பு HVX_OP_MUL_F16(vec1, vec2_inv);
#இல்லையெனில்
    திரும்பு hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_mask, vec_hf_one_1_0);
#endif
}

#define hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    செய் {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src0_type * restrict vsrc0 = (src0_type *) src0;                                  \
        src1_type * restrict vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        மாறிலி HVX_Vector f32_nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                    \
        மாறிலி HVX_Vector f16_nan_inf_mask = Q6_Vh_vsplat_R(0x7c00);                       \
        மாறிலி HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        மாறிலி uint32_t nvec = n / VLEN_FP16;                                              \
        மாறிலி uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        க்கு (; i < nvec; i++) {                                                           \
            HVX_Vector res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_mask, f16_nan_inf_mask,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        எனில் (nloe) {                                                                       \
            HVX_Vector res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_mask, f16_nan_inf_mask,   \
                                                    hf_one);                              \
            vec_store((வெறுமை *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } போது(0)

#define hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    செய் {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src0_type * restrict vsrc0 = (src0_type *) src0;                             \
        src1_type * restrict vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        மாறிலி HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        மாறிலி uint32_t nvec = n / VLEN_FP32;                                         \
        மாறிலி uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        க்கு (; i < nvec; i++) {                                                      \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        எனில் (nloe) {                                                                  \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((வெறுமை *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } போது(0)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
நிலையான உள்வரி வெறுமை OP_NAME##_aaa(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    ப்ரதிஜ்ஞா((uintptr_t) dst % 128 == 0);                                                                                              \
    ப்ரதிஜ்ஞா((uintptr_t) src0 % 128 == 0);                                                                                             \
    ப்ரதிஜ்ஞா((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
நிலையான உள்வரி வெறுமை OP_NAME##_aau(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    ப்ரதிஜ்ஞா((uintptr_t) dst % 128 == 0);                                                                                              \
    ப்ரதிஜ்ஞா((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
நிலையான உள்வரி வெறுமை OP_NAME##_aua(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    ப்ரதிஜ்ஞா((uintptr_t) dst % 128 == 0);                                                                                              \
    ப்ரதிஜ்ஞா((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_Vector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
நிலையான உள்வரி வெறுமை OP_NAME##_auu(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    ப்ரதிஜ்ஞா((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
நிலையான உள்வரி வெறுமை OP_NAME##_uaa(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    ப்ரதிஜ்ஞா((uintptr_t) src0 % 128 == 0);                                                                                             \
    ப்ரதிஜ்ஞா((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
நிலையான உள்வரி வெறுமை OP_NAME##_uau(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    ப்ரதிஜ்ஞா((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
நிலையான உள்வரி வெறுமை OP_NAME##_uua(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    ப்ரதிஜ்ஞா((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_Vector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
நிலையான உள்வரி வெறுமை OP_NAME##_uuu(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#define HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
நிலையான உள்வரி வெறுமை OP_NAME(uint8_t * restrict dst, மாறிலி uint8_t * restrict src0, மாறிலி uint8_t * restrict src1, மாறிலி uint32_t இலக்கம்_elems) { \
    எனில் (hex_is_aligned((வெறுமை *) dst, 128)) {                                                                                                 \
        எனில் (hex_is_aligned((வெறுமை *) src0, 128)) {                                                                                            \
            எனில் (hex_is_aligned((வெறுமை *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, இலக்கம்_elems);                                               \
            இல்லையெனில்                                    OP_NAME##_aau(dst, src0, src1, இலக்கம்_elems);                                               \
        } இல்லையெனில் {                                                                                                                             \
            எனில் (hex_is_aligned((வெறுமை *) src1, 128)) OP_NAME##_aua(dst, src0, src1, இலக்கம்_elems);                                               \
            இல்லையெனில்                                    OP_NAME##_auu(dst, src0, src1, இலக்கம்_elems);                                               \
        }                                                                                                                                    \
    } இல்லையெனில் {                                                                                                                                 \
        எனில் (hex_is_aligned((வெறுமை *) src0, 128)) {                                                                                            \
            எனில் (hex_is_aligned((வெறுமை *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, இலக்கம்_elems);                                               \
            இல்லையெனில்                                    OP_NAME##_uau(dst, src0, src1, இலக்கம்_elems);                                               \
        } இல்லையெனில் {                                                                                                                             \
            எனில் (hex_is_aligned((வெறுமை *) src1, 128)) OP_NAME##_uua(dst, src0, src1, இலக்கம்_elems);                                               \
            இல்லையெனில்                                    OP_NAME##_uuu(dst, src0, src1, இலக்கம்_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#undef HVX_OP_MUL_F32
#undef HVX_OP_MUL_F16

#endif // HVX_DIV_H
