#ಅನ್ತರ್ಭಾವ "binary-ops.h"

#ಆದರೆ defined(GGML_USE_ACCELERATE)
#ಅನ್ತರ್ಭಾವ <Accelerate/Accelerate.h>

ಬಳಸಿ vDSP_fn_t = ಶೂನ್ಯ (*)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *, vDSP_Stride, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *, vDSP_Stride, ದಶಮಾಂಶ *, vDSP_Stride, v_DSP_ಉದ್ದಳತೆ);
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_ಕೂಡಿಸು(ದಶಮಾಂಶ a, ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a + b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_sub(ದಶಮಾಂಶ a, ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a - b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_mul(ದಶಮಾಂಶ a, ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a * b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_div(ದಶಮಾಂಶ a, ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a / b;
}

ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ, ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_ದ್ವಿಮಾನ_op_contiguous(ಸ್ಥಿರಾಂಕ int64_t n, dst_t * z, ಸ್ಥಿರಾಂಕ src0_t * x, ಸ್ಥಿರಾಂಕ src1_t * y) {
    constexpr ಸ್ವಯಂಚಾಲಿತ src0_ಗಮ್ಯ_f32 = type_conversion_ಕೋಷ್ಟಕ<src0_t>::ಗಮ್ಯ_f32;
    constexpr ಸ್ವಯಂಚಾಲಿತ src1_ಗಮ್ಯ_f32 = type_conversion_ಕೋಷ್ಟಕ<src1_t>::ಗಮ್ಯ_f32;
    constexpr ಸ್ವಯಂಚಾಲಿತ f32_ಗಮ್ಯ_dst  = type_conversion_ಕೋಷ್ಟಕ<dst_t >::from_f32;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        z[i] = f32_ಗಮ್ಯ_dst(op(src0_ಗಮ್ಯ_f32(x[i]), src1_ಗಮ್ಯ_f32(y[i])));
    }
}

ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ, ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_ದ್ವಿಮಾನ_op_non_contiguous(ಸ್ಥಿರಾಂಕ int64_t n, ಸ್ಥಿರಾಂಕ int64_t ne10, ಸ್ಥಿರಾಂಕ int64_t nb10, dst_t * z, ಸ್ಥಿರಾಂಕ src0_t * x, ಸ್ಥಿರಾಂಕ src1_t * y) {
    constexpr ಸ್ವಯಂಚಾಲಿತ src0_ಗಮ್ಯ_f32 = type_conversion_ಕೋಷ್ಟಕ<src0_t>::ಗಮ್ಯ_f32;
    constexpr ಸ್ವಯಂಚಾಲಿತ src1_ಗಮ್ಯ_f32 = type_conversion_ಕೋಷ್ಟಕ<src1_t>::ಗಮ್ಯ_f32;
    constexpr ಸ್ವಯಂಚಾಲಿತ f32_ಗಮ್ಯ_dst  = type_conversion_ಕೋಷ್ಟಕ<dst_t >::from_f32;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ಪೂರ್ಣಾಂಕ i10 = i % ne10;
        ಸ್ಥಿರಾಂಕ src1_t * y_ptr = (ಸ್ಥಿರಾಂಕ src1_t *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)y + i10*nb10);
        z[i] = f32_ಗಮ್ಯ_dst(op(src0_ಗಮ್ಯ_f32(x[i]), src1_ಗಮ್ಯ_f32(*y_ptr)));
    }
}

ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ, ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು src1_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ apply_ದ್ವಿಮಾನ_op(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT( nb0 == ಗಾತ್ರ(dst_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(src0_t));

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, src0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_src1_contiguous_ಅಡ್ಡಸಾಲುಗಳು = ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src1);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = ಶೂನ್ಯಸೂಚಕ;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        ಆದರೆ (op == op_ಕೂಡಿಸು) {
            vDSP_op = vDSP_vadd;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ಸ್ಥಿರಾಂಕ int64_t i13 = i03 % ne13;
        ಸ್ಥಿರಾಂಕ int64_t i12 = i02 % ne12;
        ಸ್ಥಿರಾಂಕ int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((ಅಕ್ಷರ *)       dst->ದತ್ತಾಂಶ  + i03*nb3  + i02*nb2  + i01*nb1 );
        ಸ್ಥಿರಾಂಕ src0_t * src0_ptr = (ಸ್ಥಿರಾಂಕ src0_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03*nb03 + i02*nb02 + i01*nb01);
        ಸ್ಥಿರಾಂಕ src1_t * src1_ptr = (ಸ್ಥಿರಾಂಕ src1_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13*nb13 + i12*nb12 + i11*nb11);

        ಆದರೆ (is_src1_contiguous_ಅಡ್ಡಸಾಲುಗಳು) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ಸ್ಥಿರಾಂಕ int64_t nr0 = ne00 / ne10;

            ಗಾಗಿ (int64_t r = 0; r < nr0; ++r) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ACCELERATE
                ಆದರೆ constexpr (std::is_same_v<src0_t, ದಶಮಾಂಶ> && std::is_same_v<src1_t, ದಶಮಾಂಶ> && std::is_same_v<dst_t, ದಶಮಾಂಶ>) {
                    ಆದರೆ (vDSP_op != ಶೂನ್ಯಸೂಚಕ) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        ಮುಂದುವರಿ;
                    }
                }
#ಆದರೆ_ಅಂತ್ಯ
                vec_ದ್ವಿಮಾನ_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            vec_ದ್ವಿಮಾನ_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ, ದಶಮಾಂಶ)>
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_op(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    /*  */ ಆದರೆ (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_ದ್ವಿಮಾನ_op<op, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_ದ್ವಿಮಾನ_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_ದ್ವಿಮಾನ_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_bf16_t, ದಶಮಾಂಶ, ggml_bf16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_bf16_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_fp16_t, ದಶಮಾಂಶ, ggml_fp16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_fp16_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_non_quantized(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_ಕೂಡಿಸು>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sub(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_sub>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_mul(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_mul>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_div(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_div>(params, dst);
}
