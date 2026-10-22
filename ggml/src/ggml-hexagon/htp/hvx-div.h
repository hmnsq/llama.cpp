#ifndef HVX_DIV_H
#define HVX_DIV_H

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>

#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

#ଅନ୍ତର୍ଭାଵ "hvx-base.h"
#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "hvx-inverse.h"
#ଅନ୍ତର୍ଭାଵ "hvx-arith.h"

#ଯଦି __HVX_ARCH__ < 79
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#ଅନ୍ୟଥା
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#endif

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_div_mul_f16_const_using_f32(HVX_Vector vec1_hf, HVX_Vector vec2_sf_const, HVX_Vector vec_hf_one_1_0) {
#ଯଦି __HVX_ARCH__ < 79
    HVX_VectorPair src_to_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_Vector src_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_to_f32));
    HVX_Vector src_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_to_f32));
#ଅନ୍ୟଥା
    HVX_VectorPair src_to_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_Vector src_to_f32_0 = Q6_V_lo_W(src_to_f32);
    HVX_Vector src_to_f32_1 = Q6_V_hi_W(src_to_f32);
#endif

    HVX_Vector div_f32_0 = HVX_OP_MUL_F32(src_to_f32_0, vec2_sf_const);
    HVX_Vector div_f32_1 = HVX_OP_MUL_F32(src_to_f32_1, vec2_sf_const);

#ଯଦି __HVX_ARCH__ < 79
    HVX_Vector res = hvx_vec_f32_to_f16(div_f32_0, div_f32_1);
#ଅନ୍ୟଥା
    HVX_Vector res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#endif
    ଫେରନ୍ତୁ res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_div_mul_f16_const_using_f16(HVX_Vector vec1_hf, HVX_Vector const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    ଫେରନ୍ତୁ HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#define hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    କର {                                                                                               \
        dst_type * restrict vdst = (dst_type *) dst;                                                   \
        src_type * restrict vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        ଅଚଳ uint32_t nvec = n / VLEN_FP16;                                                           \
        ଅଚଳ uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        ପାଇଁ (; i < nvec; i++) {                                                                        \
            HVX_Vector res;                                                                            \
            ଯଦି (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } ଅନ୍ୟଥା {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        ଯଦି (nloe) {                                                                                    \
            HVX_Vector res;                                                                            \
            ଯଦି (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } ଅନ୍ୟଥା {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } ଯେତେବେଳେ(0)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_div_scalar_f16_aa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((ଦଶମିକ)val));
    ଅଚଳ HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);
    ପ୍ରତିଜ୍ଞା((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_div_scalar_f16_au(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((ଦଶମିକ)val));
    ଅଚଳ HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_div_scalar_f16_ua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((ଦଶମିକ)val));
    ଅଚଳ HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    ପ୍ରତିଜ୍ଞା((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_div_scalar_f16_uu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src, ଅଚଳ _Float16 val, uint32_t n) {
    ଅଚଳ HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((ଦଶମିକ)val));
    ଅଚଳ HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_vec_div_f16_using_f32(HVX_Vector vec1, HVX_Vector vec2, HVX_Vector f32_nan_inf_mask, HVX_Vector vec_hf_one_1_0) {
#ଯଦି __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_VectorPair vec1_to_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec1_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_to_f32));
    HVX_Vector     vec1_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_to_f32));

    // Convert second input to fp32
    HVX_VectorPair vec2_to_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec2_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_to_f32));
    HVX_Vector     vec2_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_to_f32));
#ଅନ୍ୟଥା
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
#ଯଦି __HVX_ARCH__ < 79
    HVX_Vector     recip = hvx_vec_f32_to_f16(div_f32_0, div_f32_1);
#ଅନ୍ୟଥା
    HVX_Vector     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#endif

    ଫେରନ୍ତୁ recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_vec_hybrid_div_f16(HVX_Vector vec1, HVX_Vector vec2, HVX_Vector f32_nan_inf_mask, HVX_Vector f16_nan_inf_mask, HVX_Vector vec_hf_one_1_0) {
#ଯଦି __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_Vector vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_mask);
    ଫେରନ୍ତୁ HVX_OP_MUL_F16(vec1, vec2_inv);
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_mask, vec_hf_one_1_0);
#endif
}

#define hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    କର {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src0_type * restrict vsrc0 = (src0_type *) src0;                                  \
        src1_type * restrict vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        ଅଚଳ HVX_Vector f32_nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                    \
        ଅଚଳ HVX_Vector f16_nan_inf_mask = Q6_Vh_vsplat_R(0x7c00);                       \
        ଅଚଳ HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        ଅଚଳ uint32_t nvec = n / VLEN_FP16;                                              \
        ଅଚଳ uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        ପାଇଁ (; i < nvec; i++) {                                                           \
            HVX_Vector res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_mask, f16_nan_inf_mask,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        ଯଦି (nloe) {                                                                       \
            HVX_Vector res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_mask, f16_nan_inf_mask,   \
                                                    hf_one);                              \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } ଯେତେବେଳେ(0)

#define hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    କର {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src0_type * restrict vsrc0 = (src0_type *) src0;                             \
        src1_type * restrict vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        ଅଚଳ HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        ଅଚଳ uint32_t nvec = n / VLEN_FP32;                                         \
        ଅଚଳ uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        ପାଇଁ (; i < nvec; i++) {                                                      \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        ଯଦି (nloe) {                                                                  \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((ଶୂନ୍ୟ *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } ଯେତେବେଳେ(0)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_aaa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                                              \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                                             \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_aau(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                                              \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_aua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                                              \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_Vector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_auu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    ପ୍ରତିଜ୍ଞା((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uaa(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                                             \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uau(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uua(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    ପ୍ରତିଜ୍ଞା((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_Vector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME##_uuu(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#define HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ OP_NAME(uint8_t * restrict dst, ଅଚଳ uint8_t * restrict src0, ଅଚଳ uint8_t * restrict src1, ଅଚଳ uint32_t ଅଙ୍କ_elems) { \
    ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) dst, 128)) {                                                                                                 \
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src0, 128)) {                                                                                            \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_aau(dst, src0, src1, ଅଙ୍କ_elems);                                               \
        } ଅନ୍ୟଥା {                                                                                                                             \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_auu(dst, src0, src1, ଅଙ୍କ_elems);                                               \
        }                                                                                                                                    \
    } ଅନ୍ୟଥା {                                                                                                                                 \
        ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src0, 128)) {                                                                                            \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_uau(dst, src0, src1, ଅଙ୍କ_elems);                                               \
        } ଅନ୍ୟଥା {                                                                                                                             \
            ଯଦି (hex_is_aligned((ଶୂନ୍ୟ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, ଅଙ୍କ_elems);                                               \
            ଅନ୍ୟଥା                                    OP_NAME##_uuu(dst, src0, src1, ଅଙ୍କ_elems);                                               \
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
