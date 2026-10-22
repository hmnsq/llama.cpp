#ಅನ್ತರ್ಭಾವ "binary-ops.h"

#ಯದಿ defined(GGML_USE_ACCELERATE)
#ಅನ್ತರ್ಭಾವ <Accelerate/Accelerate.h>

using vDSP_fn_t = ಶೂನ್ಯ (*)(ಧ್ರುವ ಭಿನ್ನ *, vDSP_Stride, ಧ್ರುವ ಭಿನ್ನ *, vDSP_Stride, ಭಿನ್ನ *, vDSP_Stride, vDSP_Length);
#endif

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_add(ಭಿನ್ನ a, ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a + b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_sub(ಭಿನ್ನ a, ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a - b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_mul(ಭಿನ್ನ a, ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a * b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_div(ಭಿನ್ನ a, ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a / b;
}

template <ಭಿನ್ನ (*op)(ಭಿನ್ನ, ಭಿನ್ನ), typename src0_t, typename src1_t, typename dst_t>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ vec_ದ್ವಿಮಾನ_op_contiguous(ಧ್ರುವ int64_t n, dst_t * z, ಧ್ರುವ src0_t * x, ಧ್ರುವ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <ಭಿನ್ನ (*op)(ಭಿನ್ನ, ಭಿನ್ನ), typename src0_t, typename src1_t, typename dst_t>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ vec_ದ್ವಿಮಾನ_op_non_contiguous(ಧ್ರುವ int64_t n, ಧ್ರುವ int64_t ne10, ಧ್ರುವ int64_t nb10, dst_t * z, ಧ್ರುವ src0_t * x, ಧ್ರುವ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        ಪೂರ್ಣಾಙ್ಕ i10 = i % ne10;
        ಧ್ರುವ src1_t * y_ptr = (ಧ್ರುವ src1_t *)((ಧ್ರುವ ವರ್ಣ *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <ಭಿನ್ನ (*op)(ಭಿನ್ನ, ಭಿನ್ನ), typename src0_t, typename src1_t, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ apply_ದ್ವಿಮಾನ_op(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT( nb0 == ಪರಿಮಾಣ(dst_t));
    GGML_ASSERT(nb00 == ಪರಿಮಾಣ(src0_t));

    ಧ್ರುವ auto [ir0, ir1] = get_thread_range(params, src0);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        ಯದಿ (op == op_add) {
            vDSP_op = vDSP_vadd;
        } ಅನ್ಯಥಾ ಯದಿ (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } ಅನ್ಯಥಾ ಯದಿ (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } ಅನ್ಯಥಾ ಯದಿ (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    ಪ್ರತಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಧ್ರುವ int64_t i03 = ir/(ne02*ne01);
        ಧ್ರುವ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಧ್ರುವ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ಧ್ರುವ int64_t i13 = i03 % ne13;
        ಧ್ರುವ int64_t i12 = i02 % ne12;
        ಧ್ರುವ int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((ವರ್ಣ *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ಧ್ರುವ src0_t * src0_ptr = (ಧ್ರುವ src0_t *) ((ಧ್ರುವ ವರ್ಣ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        ಧ್ರುವ src1_t * src1_ptr = (ಧ್ರುವ src1_t *) ((ಧ್ರುವ ವರ್ಣ *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        ಯದಿ (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ಧ್ರುವ int64_t nr0 = ne00 / ne10;

            ಪ್ರತಿ (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                ಯದಿ constexpr (std::is_same_v<src0_t, ಭಿನ್ನ> && std::is_same_v<src1_t, ಭಿನ್ನ> && std::is_same_v<dst_t, ಭಿನ್ನ>) {
                    ಯದಿ (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        ಅನುವರ್ತ;
                    }
                }
#endif
                vec_ದ್ವಿಮಾನ_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } ಅನ್ಯಥಾ {
            vec_ದ್ವಿಮಾನ_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ಭಿನ್ನ (*op)(ಭಿನ್ನ, ಭಿನ್ನ)>
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_op(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    /*  */ ಯದಿ (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_ದ್ವಿಮಾನ_op<op, ಭಿನ್ನ, ಭಿನ್ನ, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_ದ್ವಿಮಾನ_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_ದ್ವಿಮಾನ_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_bf16_t, ಭಿನ್ನ, ggml_bf16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_bf16_t, ಭಿನ್ನ, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_fp16_t, ಭಿನ್ನ, ggml_fp16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ದ್ವಿಮಾನ_op<op, ggml_fp16_t, ಭಿನ್ನ, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

ಶೂನ್ಯ ggml_compute_forward_add_non_quantized(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_add>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sub(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_sub>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_mul(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_mul>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_div(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ದ್ವಿಮಾನ_op<op_div>(params, dst);
}
