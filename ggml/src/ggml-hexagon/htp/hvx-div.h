#که_نه_تعریف_شوی HVX_DIV_H
#تعریف_کړه HVX_DIV_H

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
#تعریف_کړه HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#تعریف_کړه HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#وإلا
#تعریف_کړه HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#تعریف_کړه HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#د_که_پای

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
ساكن مضمن HVX_ویکټور hvx_div_mul_f16_const_using_f32(HVX_ویکټور vec1_hf, HVX_ویکټور vec2_sf_const, HVX_ویکټور vec_hf_one_1_0) {
#إذا __HVX_ARCH__ < 79
    HVX_ویکټور_Pair src_منزل_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ویکټور src_منزل_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_منزل_f32));
    HVX_ویکټور src_منزل_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_منزل_f32));
#وإلا
    HVX_ویکټور_Pair src_منزل_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_ویکټور src_منزل_f32_0 = Q6_V_lo_W(src_منزل_f32);
    HVX_ویکټور src_منزل_f32_1 = Q6_V_hi_W(src_منزل_f32);
#د_که_پای

    HVX_ویکټور div_f32_0 = HVX_OP_MUL_F32(src_منزل_f32_0, vec2_sf_const);
    HVX_ویکټور div_f32_1 = HVX_OP_MUL_F32(src_منزل_f32_1, vec2_sf_const);

#إذا __HVX_ARCH__ < 79
    HVX_ویکټور res = hvx_vec_f32_منزل_f16(div_f32_0, div_f32_1);
#وإلا
    HVX_ویکټور res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#د_که_پای
    أرجع res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
ساكن مضمن HVX_ویکټور hvx_div_mul_f16_const_using_f16(HVX_ویکټور vec1_hf, HVX_ویکټور const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    أرجع HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#تعریف_کړه hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    افعل {                                                                                               \
        dst_type * محدود vdst = (dst_type *) dst;                                                   \
        src_type * محدود vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_ویکټور hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        ثابت uint32_t nvec = n / VLEN_FP16;                                                           \
        ثابت uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        لكل (; i < nvec; i++) {                                                                        \
            HVX_ویکټور res;                                                                            \
            إذا (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } وإلا {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        إذا (nloe) {                                                                                    \
            HVX_ویکټور res;                                                                            \
            إذا (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } وإلا {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } طالما(0)

ساكن مضمن فراغ hvx_div_scalar_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ویکټور val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    أكد((uintptr_t) dst % 128 == 0);
    أكد((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ویکټور, HVX_ویکټور, hvx_vec_store_a);
}
ساكن مضمن فراغ hvx_div_scalar_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ویکټور val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    أكد((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_ویکټور, HVX_UVector, hvx_vec_store_a);
}
ساكن مضمن فراغ hvx_div_scalar_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ویکټور val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    أكد((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_ویکټور, hvx_vec_store_u);
}
ساكن مضمن فراغ hvx_div_scalar_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, uint32_t n) {
    ثابت HVX_ویکټور val_vec_f32 = hvx_vec_splat_f32(1.0f/((عائم)val));
    ثابت HVX_ویکټور val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
ساكن مضمن HVX_ویکټور hvx_vec_div_f16_using_f32(HVX_ویکټور vec1, HVX_ویکټور vec2, HVX_ویکټور f32_nan_inf_ماسک, HVX_ویکټور vec_hf_one_1_0) {
#إذا __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_ویکټور_Pair vec1_منزل_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ویکټور     vec1_منزل_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_منزل_f32));
    HVX_ویکټور     vec1_منزل_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_منزل_f32));

    // Convert second input to fp32
    HVX_ویکټور_Pair vec2_منزل_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ویکټور     vec2_منزل_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_منزل_f32));
    HVX_ویکټور     vec2_منزل_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_منزل_f32));
#وإلا
    // Convert first input to fp32
    HVX_ویکټور_Pair vec1_منزل_f32   = Q6_Wsf_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_ویکټور     vec1_منزل_f32_0 = Q6_V_lo_W(vec1_منزل_f32);
    HVX_ویکټور     vec1_منزل_f32_1 = Q6_V_hi_W(vec1_منزل_f32);

    // Convert second input to fp32
    HVX_ویکټور_Pair vec2_منزل_f32   = Q6_Wsf_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_ویکټور     vec2_منزل_f32_0 = Q6_V_lo_W(vec2_منزل_f32);
    HVX_ویکټور     vec2_منزل_f32_1 = Q6_V_hi_W(vec2_منزل_f32);
#د_که_پای

    // Inverse second input in fp32
    HVX_ویکټور     vec2_inv_f32_0 = hvx_vec_inverse_f32_guard(vec2_منزل_f32_0, f32_nan_inf_ماسک);
    HVX_ویکټور     vec2_inv_f32_1 = hvx_vec_inverse_f32_guard(vec2_منزل_f32_1, f32_nan_inf_ماسک);

    // Multiply first input by inverse of second, in fp32
    HVX_ویکټور     div_f32_0 = HVX_OP_MUL_F32(vec1_منزل_f32_0, vec2_inv_f32_0);
    HVX_ویکټور     div_f32_1 = HVX_OP_MUL_F32(vec1_منزل_f32_1, vec2_inv_f32_1);

    // Convert back to fp16
#إذا __HVX_ARCH__ < 79
    HVX_ویکټور     recip = hvx_vec_f32_منزل_f16(div_f32_0, div_f32_1);
#وإلا
    HVX_ویکټور     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#د_که_پای

    أرجع recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
ساكن مضمن HVX_ویکټور hvx_vec_hybrid_div_f16(HVX_ویکټور vec1, HVX_ویکټور vec2, HVX_ویکټور f32_nan_inf_ماسک, HVX_ویکټور f16_nan_inf_ماسک, HVX_ویکټور vec_hf_one_1_0) {
#إذا __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_ویکټور vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_ماسک);
    أرجع HVX_OP_MUL_F16(vec1, vec2_inv);
#وإلا
    أرجع hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_ماسک, vec_hf_one_1_0);
#د_که_پای
}

#تعریف_کړه hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    افعل {                                                                                  \
        dst_type * محدود vdst = (dst_type *) dst;                                      \
        src0_type * محدود vsrc0 = (src0_type *) src0;                                  \
        src1_type * محدود vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        ثابت HVX_ویکټور f32_nan_inf_ماسک = Q6_V_vsplat_R(0x7f800000);                    \
        ثابت HVX_ویکټور f16_nan_inf_ماسک = Q6_Vh_vsplat_R(0x7c00);                       \
        ثابت HVX_ویکټور hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        ثابت uint32_t nvec = n / VLEN_FP16;                                              \
        ثابت uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        لكل (; i < nvec; i++) {                                                           \
            HVX_ویکټور res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ماسک, f16_nan_inf_ماسک,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        إذا (nloe) {                                                                       \
            HVX_ویکټور res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_ماسک, f16_nan_inf_ماسک,   \
                                                    hf_one);                              \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } طالما(0)

#تعریف_کړه hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    افعل {                                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                                 \
        src0_type * محدود vsrc0 = (src0_type *) src0;                             \
        src1_type * محدود vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        ثابت HVX_ویکټور nan_inf_ماسک = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        ثابت uint32_t nvec = n / VLEN_FP32;                                         \
        ثابت uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        لكل (; i < nvec; i++) {                                                      \
            HVX_ویکټور inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ماسک); \
            HVX_ویکټور res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        إذا (nloe) {                                                                  \
            HVX_ویکټور inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_ماسک); \
            HVX_ویکټور res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((فراغ *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } طالما(0)

// Generic macro to define alignment permutations for an op
#تعریف_کړه DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
ساكن مضمن فراغ OP_NAME##_aaa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ویکټور, HVX_ویکټور, HVX_ویکټور, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_aau(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ویکټور, HVX_ویکټور, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_aua(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_ویکټور, HVX_UVector, HVX_ویکټور, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_auu(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_ویکټور, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uaa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ویکټور, HVX_ویکټور, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uau(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_ویکټور, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uua(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    أكد((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_ویکټور, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
ساكن مضمن فراغ OP_NAME##_uuu(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#تعریف_کړه HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
ساكن مضمن فراغ OP_NAME(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, ثابت uint32_t عدد_elems) { \
    إذا (hex_is_aligned((فراغ *) dst, 128)) {                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_aau(dst, src0, src1, عدد_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_aua(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_auu(dst, src0, src1, عدد_elems);                                               \
        }                                                                                                                                    \
    } وإلا {                                                                                                                                 \
        إذا (hex_is_aligned((فراغ *) src0, 128)) {                                                                                            \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_uau(dst, src0, src1, عدد_elems);                                               \
        } وإلا {                                                                                                                             \
            إذا (hex_is_aligned((فراغ *) src1, 128)) OP_NAME##_uua(dst, src0, src1, عدد_elems);                                               \
            وإلا                                    OP_NAME##_uuu(dst, src0, src1, عدد_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#تعریف_لغوه HVX_OP_MUL_F32
#تعریف_لغوه HVX_OP_MUL_F16

#د_که_پای // HVX_DIV_H
