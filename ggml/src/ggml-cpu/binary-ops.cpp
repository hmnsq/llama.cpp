#ਅਨ੍ਤਰ੍ਭਾਵ "binary-ops.h"

#ਜੇ defined(GGML_USE_ACCELERATE)
#ਅਨ੍ਤਰ੍ਭਾਵ <Accelerate/Accelerate.h>

ਵਰਤੋਂ vDSP_fn_t = ਖਾਲੀ (*)(ਅਚਲ ਦਸ਼ਮਲਵ *, vDSP_Stride, ਅਚਲ ਦਸ਼ਮਲਵ *, vDSP_Stride, ਦਸ਼ਮਲਵ *, vDSP_Stride, v_DSP_ਲੰਮਾਈ);
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_ਜੋੜੋ(ਦਸ਼ਮਲਵ a, ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a + b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_sub(ਦਸ਼ਮਲਵ a, ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a - b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_mul(ਦਸ਼ਮਲਵ a, ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a * b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_div(ਦਸ਼ਮਲਵ a, ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a / b;
}

ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_ਦੁਆਧਾਰੀ_op_contiguous(ਅਚਲ int64_t n, dst_t * z, ਅਚਲ src0_t * x, ਅਚਲ src1_t * y) {
    constexpr ਸਵੈਚਾਲਿਤ src0_ਮੰਜ਼ਿਲ_f32 = type_conversion_ਸਾਰਣੀ<src0_t>::ਮੰਜ਼ਿਲ_f32;
    constexpr ਸਵੈਚਾਲਿਤ src1_ਮੰਜ਼ਿਲ_f32 = type_conversion_ਸਾਰਣੀ<src1_t>::ਮੰਜ਼ਿਲ_f32;
    constexpr ਸਵੈਚਾਲਿਤ f32_ਮੰਜ਼ਿਲ_dst  = type_conversion_ਸਾਰਣੀ<dst_t >::from_f32;

    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        z[i] = f32_ਮੰਜ਼ਿਲ_dst(op(src0_ਮੰਜ਼ਿਲ_f32(x[i]), src1_ਮੰਜ਼ਿਲ_f32(y[i])));
    }
}

ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_ਦੁਆਧਾਰੀ_op_non_contiguous(ਅਚਲ int64_t n, ਅਚਲ int64_t ne10, ਅਚਲ int64_t nb10, dst_t * z, ਅਚਲ src0_t * x, ਅਚਲ src1_t * y) {
    constexpr ਸਵੈਚਾਲਿਤ src0_ਮੰਜ਼ਿਲ_f32 = type_conversion_ਸਾਰਣੀ<src0_t>::ਮੰਜ਼ਿਲ_f32;
    constexpr ਸਵੈਚਾਲਿਤ src1_ਮੰਜ਼ਿਲ_f32 = type_conversion_ਸਾਰਣੀ<src1_t>::ਮੰਜ਼ਿਲ_f32;
    constexpr ਸਵੈਚਾਲਿਤ f32_ਮੰਜ਼ਿਲ_dst  = type_conversion_ਸਾਰਣੀ<dst_t >::from_f32;

    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ਪੂਰਨ i10 = i % ne10;
        ਅਚਲ src1_t * y_ptr = (ਅਚਲ src1_t *)((ਅਚਲ ਅੱਖਰ *)y + i10*nb10);
        z[i] = f32_ਮੰਜ਼ਿਲ_dst(op(src0_ਮੰਜ਼ਿਲ_f32(x[i]), src1_ਮੰਜ਼ਿਲ_f32(*y_ptr)));
    }
}

ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ src1_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ apply_ਦੁਆਧਾਰੀ_op(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    GGML_ASSERT( nb0 == ਆਕਾਰ(dst_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(src0_t));

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, src0);
    ਅਚਲ ਤਾਰਕਿਕ is_src1_contiguous_ਪੰਕਤੀਆਂ = ggml_is_contiguous_ਪੰਕਤੀਆਂ(src1);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = ਖਾਲੀ_ਸੂਚਕ;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        ਜੇ (op == op_ਜੋੜੋ) {
            vDSP_op = vDSP_vadd;
        } ਨਹੀਂਤਾਂ ਜੇ (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } ਨਹੀਂਤਾਂ ਜੇ (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } ਨਹੀਂਤਾਂ ਜੇ (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#ਜੇ_ਅੰਤ

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne02*ne01);
        ਅਚਲ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ਅਚਲ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ਅਚਲ int64_t i13 = i03 % ne13;
        ਅਚਲ int64_t i12 = i02 % ne12;
        ਅਚਲ int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((ਅੱਖਰ *)       dst->ਅੰਕੜੇ  + i03*nb3  + i02*nb2  + i01*nb1 );
        ਅਚਲ src0_t * src0_ptr = (ਅਚਲ src0_t *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03*nb03 + i02*nb02 + i01*nb01);
        ਅਚਲ src1_t * src1_ptr = (ਅਚਲ src1_t *) ((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13*nb13 + i12*nb12 + i11*nb11);

        ਜੇ (is_src1_contiguous_ਪੰਕਤੀਆਂ) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ਅਚਲ int64_t nr0 = ne00 / ne10;

            ਲਈ (int64_t r = 0; r < nr0; ++r) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ACCELERATE
                ਜੇ constexpr (std::is_same_v<src0_t, ਦਸ਼ਮਲਵ> && std::is_same_v<src1_t, ਦਸ਼ਮਲਵ> && std::is_same_v<dst_t, ਦਸ਼ਮਲਵ>) {
                    ਜੇ (vDSP_op != ਖਾਲੀ_ਸੂਚਕ) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        ਜਾਰੀ;
                    }
                }
#ਜੇ_ਅੰਤ
                vec_ਦੁਆਧਾਰੀ_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } ਨਹੀਂਤਾਂ {
            vec_ਦੁਆਧਾਰੀ_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ)>
ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_op(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    /*  */ ਜੇ (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_ਦੁਆਧਾਰੀ_op<op, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_ਦੁਆਧਾਰੀ_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_ਦੁਆਧਾਰੀ_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_ਦੁਆਧਾਰੀ_op<op, ggml_bf16_t, ਦਸ਼ਮਲਵ, ggml_bf16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ਦੁਆਧਾਰੀ_op<op, ggml_bf16_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_ਦੁਆਧਾਰੀ_op<op, ggml_fp16_t, ਦਸ਼ਮਲਵ, ggml_fp16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ਦੁਆਧਾਰੀ_op<op, ggml_fp16_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_non_quantized(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਦੁਆਧਾਰੀ_op<op_ਜੋੜੋ>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_sub(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਦੁਆਧਾਰੀ_op<op_sub>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_mul(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਦੁਆਧਾਰੀ_op<op_mul>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_div(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਦੁਆਧਾਰੀ_op<op_div>(params, dst);
}
