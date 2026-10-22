#अन्तर्भाव "unary-ops.h"

स्थिर अन्तःस्थ भिन्न op_abs(भिन्न x) {
    पटिनिवत्त fabsf(x);
}

स्थिर अन्तःस्थ भिन्न op_sgn(भिन्न x) {
    पटिनिवत्त (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

स्थिर अन्तःस्थ भिन्न op_neg(भिन्न x) {
    पटिनिवत्त -x;
}

स्थिर अन्तःस्थ भिन्न op_step(भिन्न x) {
    पटिनिवत्त (x > 0.f) ? 1.f : 0.f;
}

स्थिर अन्तःस्थ भिन्न op_tanh(भिन्न x) {
    पटिनिवत्त tanhf(x);
}

स्थिर अन्तःस्थ भिन्न op_elu(भिन्न x) {
    पटिनिवत्त (x > 0.f) ? x : expm1f(x);
}

स्थिर अन्तःस्थ भिन्न op_relu(भिन्न x) {
    पटिनिवत्त (x > 0.f) ? x : 0.f;
}

स्थिर अन्तःस्थ भिन्न op_sigmoid(भिन्न x) {
    पटिनिवत्त 1.f / (1.f + expf(-x));
}

स्थिर अन्तःस्थ भिन्न op_hardsigmoid(भिन्न x) {
    पटिनिवत्त fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

स्थिर अन्तःस्थ भिन्न op_exp(भिन्न x) {
    पटिनिवत्त expf(x);
}

स्थिर अन्तःस्थ भिन्न op_hardswish(भिन्न x) {
    पटिनिवत्त x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

स्थिर अन्तःस्थ भिन्न op_sqr(भिन्न x) {
    पटिनिवत्त x * x;
}

स्थिर अन्तःस्थ भिन्न op_sqrt(भिन्न x) {
    पटिनिवत्त sqrtf(x);
}

स्थिर अन्तःस्थ भिन्न op_xielu(भिन्न x, भिन्न alpha_n, भिन्न alpha_p, भिन्न beta, भिन्न eps) {
    सचे (x > 0.0f) {
        पटिनिवत्त alpha_p * x * x + beta * x;
    } अञ्ञथा {
        ध्रुव भिन्न min_x_eps = fminf(x, eps);
        पटिनिवत्त (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

स्थिर अन्तःस्थ भिन्न op_sin(भिन्न x) {
    पटिनिवत्त sinf(x);
}

स्थिर अन्तःस्थ भिन्न op_cos(भिन्न x) {
    पटिनिवत्त cosf(x);
}

स्थिर अन्तःस्थ भिन्न op_log(भिन्न x) {
    पटिनिवत्त logf(x);
}

स्थिर अन्तःस्थ भिन्न op_expm1(भिन्न x) {
    पटिनिवत्त expf(x) - 1.0f;
}

स्थिर अन्तःस्थ भिन्न op_softplus(भिन्न x) {
    पटिनिवत्त (x > 20.0f) ? x : logf(1.0f + expf(x));
}

स्थिर अन्तःस्थ भिन्न op_floor(भिन्न x) {
    पटिनिवत्त floorf(x);
}

स्थिर अन्तःस्थ भिन्न op_ceil(भिन्न x) {
    पटिनिवत्त ceilf(x);
}

स्थिर अन्तःस्थ भिन्न op_round(भिन्न x) {
    पटिनिवत्त roundf(x);
}

स्थिर अन्तःस्थ भिन्न op_trunc(भिन्न x) {
    पटिनिवत्त truncf(x);
}

template <भिन्न (*op)(भिन्न), typename src0_t, typename dst_t>
स्थिर अन्तःस्थ शून्य vec_unary_op(int64_t n, dst_t * y, ध्रुव src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    अत्थाय (पूर्णाङ्क i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <भिन्न (*op)(भिन्न), typename src0_t, typename dst_t>
स्थिर शून्य apply_unary_op(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(dst_t));
    GGML_ASSERT(nb00 == परिमाण(src0_t));

    ध्रुव auto [ir0, ir1] = get_thread_range(params, src0);

    अत्थाय (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((वर्ण *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ध्रुव src0_t * src0_ptr = (ध्रुव src0_t *) ((ध्रुव वर्ण *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <भिन्न (*op)(भिन्न)>
स्थिर शून्य unary_op(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    /*  */ सचे (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, भिन्न, भिन्न>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, भिन्न>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, भिन्न>(params, dst);
    } अञ्ञथा {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <भिन्न (*op)(भिन्न, ggml_tensor *)>
स्थिर शून्य unary_op_params(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    /*  */ सचे (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, भिन्न, भिन्न>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, भिन्न>(params, dst);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, भिन्न>(params, dst);
    } अञ्ञथा {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
स्थिर अन्तःस्थ शून्य vec_unary_op_functor(int64_t n, dst_t * y, ध्रुव src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    अत्थाय (पूर्णाङ्क i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
स्थिर शून्य apply_unary_op_functor(ध्रुव ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(dst_t));
    GGML_ASSERT(nb00 == परिमाण(src0_t));

    ध्रुव auto [ir0, ir1] = get_thread_range(params, src0);

    अत्थाय (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((वर्ण *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ध्रुव src0_t * src0_ptr = (ध्रुव src0_t *) ((ध्रुव वर्ण *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
स्थिर शून्य unary_op_functor(ध्रुव ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    /*  */ सचे (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, भिन्न, भिन्न>(params, dst, op);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, भिन्न>(params, dst, op);
    } अञ्ञथा सचे (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, भिन्न>(params, dst, op);
    } अञ्ञथा {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

शून्य ggml_compute_forward_abs(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

शून्य ggml_compute_forward_sgn(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

शून्य ggml_compute_forward_neg(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

शून्य ggml_compute_forward_step(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

शून्य ggml_compute_forward_tanh(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

शून्य ggml_compute_forward_elu(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

शून्य ggml_compute_forward_relu(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

शून्य ggml_compute_forward_sigmoid(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

शून्य ggml_compute_forward_hardsigmoid(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

शून्य ggml_compute_forward_exp(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

शून्य ggml_compute_forward_hardswish(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

शून्य ggml_compute_forward_sqr(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

शून्य ggml_compute_forward_sqrt(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

शून्य ggml_compute_forward_sin(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

शून्य ggml_compute_forward_cos(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

शून्य ggml_compute_forward_log(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

शून्य ggml_compute_forward_expm1(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

शून्य ggml_compute_forward_softplus(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

शून्य ggml_compute_forward_floor(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

शून्य ggml_compute_forward_ceil(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

शून्य ggml_compute_forward_round(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

शून्य ggml_compute_forward_trunc(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

शून्य ggml_compute_forward_xielu(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव भिन्न alpha_n = ggml_get_op_params_f32(dst, 1);
    ध्रुव भिन्न alpha_p = ggml_get_op_params_f32(dst, 2);
    ध्रुव भिन्न beta = ggml_get_op_params_f32(dst, 3);
    ध्रुव भिन्न eps = ggml_get_op_params_f32(dst, 4);

    ध्रुव auto xielu_op_params = [alpha_n, alpha_p, beta, eps](भिन्न f) {
        पटिनिवत्त op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

