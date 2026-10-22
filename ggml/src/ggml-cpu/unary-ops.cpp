#ಅನ್ತರ್ಭಾವ "unary-ops.h"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_abs(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ fabsf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_sgn(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_neg(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ -x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_ಹೆಜ್ಜೆ(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (x > 0.f) ? 1.f : 0.f;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_tanh(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ tanhf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_elu(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (x > 0.f) ? x : expm1f(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_relu(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (x > 0.f) ? x : 0.f;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_sigmoid(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ 1.f / (1.f + expf(-x));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_hardsigmoid(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_exp(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ expf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_hardswish(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_sqr(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ x * x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_sqrt(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ sqrtf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_xielu(ದಶಮಾಂಶ x, ದಶಮಾಂಶ alpha_n, ದಶಮಾಂಶ alpha_p, ದಶಮಾಂಶ beta, ದಶಮಾಂಶ eps) {
    ಆದರೆ (x > 0.0f) {
        ಹಿಂತಿರುಗಿ alpha_p * x * x + beta * x;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min_x_eps = fminf(x, eps);
        ಹಿಂತಿರುಗಿ (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_sin(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ sinf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_cos(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ cosf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_log(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ logf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_expm1(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ expf(x) - 1.0f;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_softplus(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (x > 20.0f) ? x : logf(1.0f + expf(x));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_floor(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ floorf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_ceil(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ ceilf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_ಸುತ್ತು(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ roundf(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ op_trunc(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ truncf(x);
}

ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_unary_op(int64_t n, dst_t * y, ಸ್ಥಿರಾಂಕ src0_t * x) {
    constexpr ಸ್ವಯಂಚಾಲಿತ src0_ಗಮ್ಯ_f32 = type_conversion_ಕೋಷ್ಟಕ<src0_t>::ಗಮ್ಯ_f32;
    constexpr ಸ್ವಯಂಚಾಲಿತ f32_ಗಮ್ಯ_dst  = type_conversion_ಕೋಷ್ಟಕ<dst_t >::from_f32;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        y[i] = f32_ಗಮ್ಯ_dst(op(src0_ಗಮ್ಯ_f32(x[i])));
    }
}

ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ), ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ apply_unary_op(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0) && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಗಾತ್ರ(dst_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(src0_t));

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, src0);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ಅಕ್ಷರ *)       dst->ದತ್ತಾಂಶ  + i03*nb3  + i02*nb2  + i01*nb1 );
        ಸ್ಥಿರಾಂಕ src0_t * src0_ptr = (ಸ್ಥಿರಾಂಕ src0_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ)>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    /*  */ ಆದರೆ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ಮಾದರಿ <ದಶಮಾಂಶ (*op)(ದಶಮಾಂಶ, ggml_tensor *)>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_params(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    /*  */ ಆದರೆ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ದಶಮಾಂಶ>(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Op, ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_unary_op_functor(int64_t n, dst_t * y, ಸ್ಥಿರಾಂಕ src0_t * x, Op op) {
    constexpr ಸ್ವಯಂಚಾಲಿತ src0_ಗಮ್ಯ_f32 = type_conversion_ಕೋಷ್ಟಕ<src0_t>::ಗಮ್ಯ_f32;
    constexpr ಸ್ವಯಂಚಾಲಿತ f32_ಗಮ್ಯ_dst  = type_conversion_ಕೋಷ್ಟಕ<dst_t >::from_f32;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        y[i] = f32_ಗಮ್ಯ_dst(op(src0_ಗಮ್ಯ_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Op, ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ apply_unary_op_functor(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಗಾತ್ರ(dst_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(src0_t));

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, src0);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ಅಕ್ಷರ *)       dst->ದತ್ತಾಂಶ  + i03*nb3  + i02*nb2  + i01*nb1 );
        ಸ್ಥಿರಾಂಕ src0_t * src0_ptr = (ಸ್ಥಿರಾಂಕ src0_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Op>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_functor(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    /*  */ ಆದರೆ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(params, dst, op);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, ದಶಮಾಂಶ>(params, dst, op);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, ದಶಮಾಂಶ>(params, dst, op);
    } ಇಲ್ಲದಿದ್ದರೆ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ಶೂನ್ಯ ggml_compute_forward_abs(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sgn(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_neg(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_ಹೆಜ್ಜೆ(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ಹೆಜ್ಜೆ>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_tanh(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_elu(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_relu(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sigmoid(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_hardsigmoid(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_exp(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_hardswish(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sqr(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sqrt(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sin(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_cos(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_log(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_expm1(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_softplus(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_floor(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_ceil(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_ಸುತ್ತು(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ಸುತ್ತು>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_trunc(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_xielu(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha_n = ggml_get_op_params_f32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha_p = ggml_get_op_params_f32(dst, 2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta = ggml_get_op_params_f32(dst, 3);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps = ggml_get_op_params_f32(dst, 4);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ xielu_op_params = [alpha_n, alpha_p, beta, eps](ದಶಮಾಂಶ f) {
        ಹಿಂತಿರುಗಿ op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

