#جيڪڏهن_وصف_نه_ٿيل HVX_DIV_H
#وصف_ڪريو HVX_DIV_H

#ضمن <HAP_farf.h>

#ضمن <math.h>
#ضمن <string.h>
#ضمن <assert.h>
#ضمن <stddef.h>
#ضمن <stdint.h>

#ضمن "hvx-base.h"
#ضمن "hex-utils.h"
#ضمن "hvx-inverse.h"
#ضمن "hvx-arith.h"

#إذا __HVX_ARCH__ < 79
#وصف_ڪريو HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#وصف_ڪريو HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#وإلا
#وصف_ڪريو HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#وصف_ڪريو HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#جيڪڏهن_پورو

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
ساكن مضمن HVX_ويڪٽر hvx_div_mul_f16_const_using_f32(HVX_ويڪٽر vec1_hf, HVX_ويڪٽر vec2_sf_const, HVX_ويڪٽر vec_hf_one_1_0) {
#إذا __HVX_ARCH__ < 79
    HVX_ويڪٽر_Pair src_منزل_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ويڪٽر src_منزل_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_منزل_f32));
    HVX_ويڪٽر src_منزل_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_منزل_f32));
#وإلا
    HVX_ويڪٽر_Pair src_منزل_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ويڪٽر src_منزل_f32_0 = Q6_V_lo_W(src_منزل_f32);
    HVX_ويڪٽر src_منزل_f32_1 = Q6_V_hi_W(src_منزل_f32);
#جيڪڏهن_پورو

    HVX_ويڪٽر div_f32_0 = HVX_OP_MUL_F32(src_منزل_f32_0, vec2_sf_const);
    HVX_ويڪٽر div_f32_1 = HVX_OP_MUL_F32(src_منزل_f32_1, vec2_sf_const);

#إذا __HVX_ARCH__ < 79
    HVX_ويڪٽر res = hvx_vec_f32_منزل_f16(div_f32_0, div_f32_1);
#وإلا
    HVX_ويڪٽر res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#جيڪڏهن_پورو
    أرجع res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
ساكن مضمن HVX_ويڪٽر hvx_div_mul_f16_const_using_f16(HVX_ويڪٽر vec1_hf, HVX_ويڪٽر const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    أرجع HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#وصف_ڪريو hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    افعل {                                                                                               \
        dst_type * محدود vdst = (dst_type *) dst;                                                   \
        src_type * محدود vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_ويڪٽر hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        ثابت uint32_t nvec = n / VLEN_FP16;                                                           \
        ثابت uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        لكل (; i < nvec; i++) {                                                                        \
            HVX_ويڪٽر res;                                                                            \
            إذا (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } وإلا {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        إذا (nloe) {                                                                                    \
            HVX_ويڪٽر res;                                                                            \
            إذا (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } وإلا {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } طالما(0)

ساكن مضمن فراغ hvx_div_scalar_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ويڪٽر val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ويڪٽر val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    أكد((uintptr_t) dst % 128 == 0);
    أكد((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ويڪٽر, HVX_ويڪٽر, hvx_vec_store_a);
}
ساكن مضمن فراغ hvx_div_scalar_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ويڪٽر val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ويڪٽر val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    أكد((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ويڪٽر, HVX_UVector, hvx_vec_store_a);
}
ساكن مضمن فراغ hvx_div_scalar_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ويڪٽر val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ويڪٽر val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    أكد((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_ويڪٽر, hvx_vec_store_u);
}
ساكن مضمن فراغ hvx_div_scalar_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ويڪٽر val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ويڪٽر val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
ساكن مضمن HVX_ويڪٽر hvx_vec_div_f16_using_f32(HVX_ويڪٽر vec1, HVX_ويڪٽر vec2, HVX_ويڪٽر f32_nan_inf_نقاب, HVX_ويڪٽر vec_hf_one_1_0) {
#إذا __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_ويڪٽر_Pair vec1_منزل_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ويڪٽر     vec1_منزل_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_منزل_f32));
    HVX_ويڪٽر     vec1_منزل_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_منزل_f32));

    // Convert second input to fp32
    HVX_ويڪٽر_Pair vec2_منزل_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ويڪٽر     vec2_منزل_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_منزل_f32));
    HVX_ويڪٽر     vec2_منزل_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_منزل_f32));
#وإلا
    // Convert first input to fp32
    HVX_ويڪٽر_Pair vec1_منزل_f32   = Q6_Wsf_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ويڪٽر     vec1_منزل_f32_0 = Q6_V_lo_W(vec1_منزل_f32);
    HVX_ويڪٽر     vec1_منزل_f32_1 = Q6_V_hi_W(vec1_منزل_f32);

    // Convert second input to fp32
    HVX_ويڪٽر_Pair vec2_منزل_f32   = Q6_Wsf_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ويڪٽر     vec2_منزل_f32_0 = Q6_V_lo_W(vec2_منزل_f32);
    HVX_ويڪٽر     vec2_منزل_f32_1 = Q6_V_hi_W(vec2_منزل_f32);
#جيڪڏهن_پورو

    // Inverse second input in fp32
    HVX_ويڪٽر     vec2_inv_f32_0 = hvx_vec_inverse_f32_guard(vec2_منزل_f32_0, f32_nan_inf_نقاب);
    HVX_ويڪٽر     vec2_inv_f32_1 = hvx_vec_inverse_f32_guard(vec2_منزل_f32_1, f32_nan_inf_نقاب);

    // Multiply first input by inverse of second, in fp32
    HVX_ويڪٽر     div_f32_0 = HVX_OP_MUL_F32(vec1_منزل_f32_0, vec2_inv_f32_0);
    HVX_ويڪٽر     div_f32_1 = HVX_OP_MUL_F32(vec1_منزل_f32_1, vec2_inv_f32_1);

    // Convert back to fp16
#إذا __HVX_ARCH__ < 79
    HVX_ويڪٽر     recip = hvx_vec_f32_منزل_f16(div_f32_0, div_f32_1);
#وإلا
    HVX_ويڪٽر     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#جيڪڏهن_پورو

    أرجع recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
ساكن مضمن HVX_ويڪٽر hvx_vec_hybrid_div_f16(HVX_ويڪٽر vec1, HVX_ويڪٽر vec2, HVX_ويڪٽر f32_nan_inf_نقاب, HVX_ويڪٽر f16_nan_inf_نقاب, HVX_ويڪٽر vec_hf_one_1_0) {
#إذا __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_ويڪٽر vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_نقاب);
    أرجع HVX_OP_MUL_F16(vec1, vec2_inv);
#وإلا
    أرجع hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_نقاب, vec_hf_one_1_0);
#جيڪڏهن_پورو
}

#وصف_ڪريو hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    افعل {                                                                                  \
        dst_type * محدود vdst = (dst_type *) dst;                                      \
        src0_type * محدود vsrc0 = (src0_type *) src0;                                  \
        src1_type * محدود vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        ثابت HVX_ويڪٽر f32_nan_inf_نقاب = Q6_V_vsplat_R(0x7f800000);                    \
        ثابت HVX_ويڪٽر f16_nan_inf_نقاب = Q6_Vh_vsplat_R(0x7c00);                       \
        ثابت HVX_ويڪٽر hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        ثابت uint32_t nvec = n / VLEN_FP16;                                              \
        ثابت uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        لكل (; i < nvec; i++) {                                                           \
            HVX_ويڪٽر res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_نقاب, f16_nan_inf_نقاب,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        إذا (nloe) {                                                                       \
            HVX_ويڪٽر res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_نقاب, f16_nan_inf_نقاب,   \
                                                    hf_one);                              \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } طالما(0)

#وصف_ڪريو hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    افعل {                                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                                 \
        src0_type * محدود vsrc0 = (src0_type *) src0;                             \
        src1_type * محدود vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        ثابت HVX_ويڪٽر nan_inf_نقاب = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        ثابت uint32_t nvec = n / VLEN_FP32;                                         \
        ثابت uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        لكل (; i < nvec; i++) {                                                      \
            HVX_ويڪٽر inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_نقاب); \
            HVX_ويڪٽر res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        إذا (nloe) {                                                                  \
            HVX_ويڪٽر inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_نقاب); \
            HVX_ويڪٽر res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } طالما(0)

// Generic macro to define alignment permutations for an op
#وصف_ڪريو DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
ساكن مضمن فراغ OP_NAME##_aaa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ويڪٽر, HVX_ويڪٽر, HVX_ويڪٽر, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_aau(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ويڪٽر, HVX_ويڪٽر, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_aua(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ويڪٽر, HVX_UVector, HVX_ويڪٽر, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_auu(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_ويڪٽر, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uaa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ويڪٽر, HVX_ويڪٽر, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uau(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ويڪٽر, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uua(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_ويڪٽر, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uuu(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#وصف_ڪريو HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
ساكن مضمن فراغ OP_NAME(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint32_t انگ_elems) { \
    إذا (hex_is_aligned((فراغ *) dst, 128)) {                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, انگ_elems);                                               \
            وإلا                                    OP_NAME##_aau(dst, src0, src1, انگ_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, انگ_elems);                                               \
            وإلا                                    OP_NAME##_auu(dst, src0, src1, انگ_elems);                                               \
        }                                                                                                                                    \
    } وإلا {                                                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, انگ_elems);                                               \
            وإلا                                    OP_NAME##_uau(dst, src0, src1, انگ_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, انگ_elems);                                               \
            وإلا                                    OP_NAME##_uuu(dst, src0, src1, انگ_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#وصف_رد HVX_OP_MUL_F32
#وصف_رد HVX_OP_MUL_F16

#جيڪڏهن_پورو // HVX_DIV_H
