#ਅਨ੍ਤਰ੍ਭਾਵ "unary-ops.h"

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_abs(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ fabsf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_sgn(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_neg(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ -x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_ਕਦਮ(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (x > 0.f) ? 1.f : 0.f;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_tanh(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ tanhf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_elu(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (x > 0.f) ? x : expm1f(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_relu(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (x > 0.f) ? x : 0.f;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_sigmoid(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ 1.f / (1.f + expf(-x));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_hardsigmoid(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_exp(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ expf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_hardswish(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_sqr(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ x * x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_sqrt(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ sqrtf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_xielu(ਦਸ਼ਮਲਵ x, ਦਸ਼ਮਲਵ alpha_n, ਦਸ਼ਮਲਵ alpha_p, ਦਸ਼ਮਲਵ beta, ਦਸ਼ਮਲਵ eps) {
    ਜੇ (x > 0.0f) {
        ਵਾਪਸ alpha_p * x * x + beta * x;
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ ਦਸ਼ਮਲਵ min_x_eps = fminf(x, eps);
        ਵਾਪਸ (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_sin(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ sinf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_cos(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ cosf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_log(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ logf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_expm1(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ expf(x) - 1.0f;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_softplus(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (x > 20.0f) ? x : logf(1.0f + expf(x));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_floor(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ floorf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_ceil(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ ceilf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_ਗੇੜਾ(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ roundf(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ op_trunc(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ truncf(x);
}

ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_unary_op(int64_t n, dst_t * y, ਅਚਲ src0_t * x) {
    constexpr ਸਵੈਚਾਲਿਤ src0_ਮੰਜ਼ਿਲ_f32 = type_conversion_ਸਾਰਣੀ<src0_t>::ਮੰਜ਼ਿਲ_f32;
    constexpr ਸਵੈਚਾਲਿਤ f32_ਮੰਜ਼ਿਲ_dst  = type_conversion_ਸਾਰਣੀ<dst_t >::from_f32;

    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        y[i] = f32_ਮੰਜ਼ਿਲ_dst(op(src0_ਮੰਜ਼ਿਲ_f32(x[i])));
    }
}

ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ), ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ apply_unary_op(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0) && ggml_is_contiguous_ਪੰਕਤੀਆਂ(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ਆਕਾਰ(dst_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(src0_t));

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, src0);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne02*ne01);
        ਅਚਲ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ਅਚਲ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ਅੱਖਰ *)       dst->ਅੰਕੜੇ  + i03*nb3  + i02*nb2  + i01*nb1 );
        ਅਚਲ src0_t * src0_ptr = (ਅਚਲ src0_t *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ)>
ਸਥਿਰ ਖਾਲੀ unary_op(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    /*  */ ਜੇ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ਸਾਂਚਾ <ਦਸ਼ਮਲਵ (*op)(ਦਸ਼ਮਲਵ, ggml_tensor *)>
ਸਥਿਰ ਖਾਲੀ unary_op_params(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    /*  */ ਜੇ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ਦਸ਼ਮਲਵ>(params, dst);
    } ਨਹੀਂਤਾਂ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Op, ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_unary_op_functor(int64_t n, dst_t * y, ਅਚਲ src0_t * x, Op op) {
    constexpr ਸਵੈਚਾਲਿਤ src0_ਮੰਜ਼ਿਲ_f32 = type_conversion_ਸਾਰਣੀ<src0_t>::ਮੰਜ਼ਿਲ_f32;
    constexpr ਸਵੈਚਾਲਿਤ f32_ਮੰਜ਼ਿਲ_dst  = type_conversion_ਸਾਰਣੀ<dst_t >::from_f32;

    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        y[i] = f32_ਮੰਜ਼ਿਲ_dst(op(src0_ਮੰਜ਼ਿਲ_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Op, ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ apply_unary_op_functor(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ਆਕਾਰ(dst_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(src0_t));

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, src0);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne02*ne01);
        ਅਚਲ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ਅਚਲ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ਅੱਖਰ *)       dst->ਅੰਕੜੇ  + i03*nb3  + i02*nb2  + i01*nb1 );
        ਅਚਲ src0_t * src0_ptr = (ਅਚਲ src0_t *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Op>
ਸਥਿਰ ਖਾਲੀ unary_op_functor(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    /*  */ ਜੇ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(params, dst, op);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, ਦਸ਼ਮਲਵ>(params, dst, op);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, ਦਸ਼ਮਲਵ>(params, dst, op);
    } ਨਹੀਂਤਾਂ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ਖਾਲੀ ggml_compute_forward_abs(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_sgn(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_neg(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_ਕਦਮ(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ਕਦਮ>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_tanh(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_elu(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_relu(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_sigmoid(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_hardsigmoid(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_exp(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_hardswish(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_sqr(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_sqrt(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_sin(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_cos(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_log(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_expm1(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_softplus(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_floor(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_ceil(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_ਗੇੜਾ(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ਗੇੜਾ>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_trunc(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

ਖਾਲੀ ggml_compute_forward_xielu(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ alpha_n = ggml_get_op_params_f32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ alpha_p = ggml_get_op_params_f32(dst, 2);
    ਅਚਲ ਦਸ਼ਮਲਵ beta = ggml_get_op_params_f32(dst, 3);
    ਅਚਲ ਦਸ਼ਮਲਵ eps = ggml_get_op_params_f32(dst, 4);

    ਅਚਲ ਸਵੈਚਾਲਿਤ xielu_op_params = [alpha_n, alpha_p, beta, eps](ਦਸ਼ਮਲਵ f) {
        ਵਾਪਸ op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

