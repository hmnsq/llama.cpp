#ಅನ್ತರ್ಭಾವ "unary-ops.h"

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_abs(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ fabsf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_sgn(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_neg(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ -x;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_step(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (x > 0.f) ? 1.f : 0.f;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_tanh(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ tanhf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_elu(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (x > 0.f) ? x : expm1f(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_relu(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (x > 0.f) ? x : 0.f;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_sigmoid(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ 1.f / (1.f + expf(-x));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_hardsigmoid(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_exp(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ expf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_hardswish(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_sqr(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ x * x;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_sqrt(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ sqrtf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_xielu(ಭಿನ್ನ x, ಭಿನ್ನ alpha_n, ಭಿನ್ನ alpha_p, ಭಿನ್ನ beta, ಭಿನ್ನ eps) {
    ಯದಿ (x > 0.0f) {
        ಪ್ರತ್ಯಾಗಮ alpha_p * x * x + beta * x;
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ ಭಿನ್ನ min_x_eps = fminf(x, eps);
        ಪ್ರತ್ಯಾಗಮ (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_sin(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ sinf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_cos(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ cosf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_log(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ logf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_expm1(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ expf(x) - 1.0f;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_softplus(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (x > 20.0f) ? x : logf(1.0f + expf(x));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_floor(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ floorf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_ceil(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ ceilf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_round(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ roundf(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ op_trunc(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ truncf(x);
}

template <ಭಿನ್ನ (*op)(ಭಿನ್ನ), typename src0_t, typename dst_t>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ vec_unary_op(int64_t n, dst_t * y, ಧ್ರುವ src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <ಭಿನ್ನ (*op)(ಭಿನ್ನ), typename src0_t, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ apply_unary_op(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಪರಿಮಾಣ(dst_t));
    GGML_ASSERT(nb00 == ಪರಿಮಾಣ(src0_t));

    ಧ್ರುವ auto [ir0, ir1] = get_thread_range(params, src0);

    ಪ್ರತಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಧ್ರುವ int64_t i03 = ir/(ne02*ne01);
        ಧ್ರುವ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಧ್ರುವ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ವರ್ಣ *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ಧ್ರುವ src0_t * src0_ptr = (ಧ್ರುವ src0_t *) ((ಧ್ರುವ ವರ್ಣ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ಭಿನ್ನ (*op)(ಭಿನ್ನ)>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    /*  */ ಯದಿ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ಭಿನ್ನ, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <ಭಿನ್ನ (*op)(ಭಿನ್ನ, ggml_tensor *)>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_params(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    /*  */ ಯದಿ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ಭಿನ್ನ, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ಭಿನ್ನ>(params, dst);
    } ಅನ್ಯಥಾ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ vec_unary_op_functor(int64_t n, dst_t * y, ಧ್ರುವ src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ apply_unary_op_functor(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಪರಿಮಾಣ(dst_t));
    GGML_ASSERT(nb00 == ಪರಿಮಾಣ(src0_t));

    ಧ್ರುವ auto [ir0, ir1] = get_thread_range(params, src0);

    ಪ್ರತಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಧ್ರುವ int64_t i03 = ir/(ne02*ne01);
        ಧ್ರುವ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಧ್ರುವ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ವರ್ಣ *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ಧ್ರುವ src0_t * src0_ptr = (ಧ್ರುವ src0_t *) ((ಧ್ರುವ ವರ್ಣ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_functor(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    /*  */ ಯದಿ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, ಭಿನ್ನ, ಭಿನ್ನ>(params, dst, op);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, ಭಿನ್ನ>(params, dst, op);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, ಭಿನ್ನ>(params, dst, op);
    } ಅನ್ಯಥಾ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ಶೂನ್ಯ ggml_compute_forward_abs(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sgn(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_neg(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_step(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_tanh(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_elu(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_relu(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sigmoid(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_hardsigmoid(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_exp(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_hardswish(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sqr(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sqrt(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_sin(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_cos(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_log(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_expm1(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_softplus(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_floor(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_ceil(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_round(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_trunc(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

ಶೂನ್ಯ ggml_compute_forward_xielu(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
    ಧ್ರುವ ಭಿನ್ನ alpha_n = ggml_get_op_params_f32(dst, 1);
    ಧ್ರುವ ಭಿನ್ನ alpha_p = ggml_get_op_params_f32(dst, 2);
    ಧ್ರುವ ಭಿನ್ನ beta = ggml_get_op_params_f32(dst, 3);
    ಧ್ರುವ ಭಿನ್ನ eps = ggml_get_op_params_f32(dst, 4);

    ಧ್ರುವ auto xielu_op_params = [alpha_n, alpha_p, beta, eps](ಭಿನ್ನ f) {
        ಪ್ರತ್ಯಾಗಮ op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

