#அந்தர்பாவ "unary-ops.h"

நிலையான உள்வரி தசம op_abs(தசம x) {
    திரும்பு fabsf(x);
}

நிலையான உள்வரி தசம op_sgn(தசம x) {
    திரும்பு (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

நிலையான உள்வரி தசம op_neg(தசம x) {
    திரும்பு -x;
}

நிலையான உள்வரி தசம op_step(தசம x) {
    திரும்பு (x > 0.f) ? 1.f : 0.f;
}

நிலையான உள்வரி தசம op_tanh(தசம x) {
    திரும்பு tanhf(x);
}

நிலையான உள்வரி தசம op_elu(தசம x) {
    திரும்பு (x > 0.f) ? x : expm1f(x);
}

நிலையான உள்வரி தசம op_relu(தசம x) {
    திரும்பு (x > 0.f) ? x : 0.f;
}

நிலையான உள்வரி தசம op_sigmoid(தசம x) {
    திரும்பு 1.f / (1.f + expf(-x));
}

நிலையான உள்வரி தசம op_hardsigmoid(தசம x) {
    திரும்பு fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

நிலையான உள்வரி தசம op_exp(தசம x) {
    திரும்பு expf(x);
}

நிலையான உள்வரி தசம op_hardswish(தசம x) {
    திரும்பு x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

நிலையான உள்வரி தசம op_sqr(தசம x) {
    திரும்பு x * x;
}

நிலையான உள்வரி தசம op_sqrt(தசம x) {
    திரும்பு sqrtf(x);
}

நிலையான உள்வரி தசம op_xielu(தசம x, தசம alpha_n, தசம alpha_p, தசம beta, தசம eps) {
    எனில் (x > 0.0f) {
        திரும்பு alpha_p * x * x + beta * x;
    } இல்லையெனில் {
        மாறிலி தசம min_x_eps = fminf(x, eps);
        திரும்பு (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

நிலையான உள்வரி தசம op_sin(தசம x) {
    திரும்பு sinf(x);
}

நிலையான உள்வரி தசம op_cos(தசம x) {
    திரும்பு cosf(x);
}

நிலையான உள்வரி தசம op_log(தசம x) {
    திரும்பு logf(x);
}

நிலையான உள்வரி தசம op_expm1(தசம x) {
    திரும்பு expf(x) - 1.0f;
}

நிலையான உள்வரி தசம op_softplus(தசம x) {
    திரும்பு (x > 20.0f) ? x : logf(1.0f + expf(x));
}

நிலையான உள்வரி தசம op_floor(தசம x) {
    திரும்பு floorf(x);
}

நிலையான உள்வரி தசம op_ceil(தசம x) {
    திரும்பு ceilf(x);
}

நிலையான உள்வரி தசம op_round(தசம x) {
    திரும்பு roundf(x);
}

நிலையான உள்வரி தசம op_trunc(தசம x) {
    திரும்பு truncf(x);
}

template <தசம (*op)(தசம), typename src0_t, typename dst_t>
நிலையான உள்வரி வெறுமை vec_unary_op(int64_t n, dst_t * y, மாறிலி src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    க்கு (முழு i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <தசம (*op)(தசம), typename src0_t, typename dst_t>
நிலையான வெறுமை apply_unary_op(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == அளவு(dst_t));
    GGML_ASSERT(nb00 == அளவு(src0_t));

    மாறிலி auto [ir0, ir1] = get_thread_range(params, src0);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne02*ne01);
        மாறிலி int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        மாறிலி int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((எழுத்து *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        மாறிலி src0_t * src0_ptr = (மாறிலி src0_t *) ((மாறிலி எழுத்து *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <தசம (*op)(தசம)>
நிலையான வெறுமை unary_op(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    /*  */ எனில் (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, தசம, தசம>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, தசம>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, தசம>(params, dst);
    } இல்லையெனில் {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <தசம (*op)(தசம, ggml_tensor *)>
நிலையான வெறுமை unary_op_params(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    /*  */ எனில் (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, தசம, தசம>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, தசம>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, தசம>(params, dst);
    } இல்லையெனில் {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
நிலையான உள்வரி வெறுமை vec_unary_op_functor(int64_t n, dst_t * y, மாறிலி src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    க்கு (முழு i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
நிலையான வெறுமை apply_unary_op_functor(மாறிலி ggml_compute_params * params, ggml_tensor * dst, Op op) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == அளவு(dst_t));
    GGML_ASSERT(nb00 == அளவு(src0_t));

    மாறிலி auto [ir0, ir1] = get_thread_range(params, src0);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne02*ne01);
        மாறிலி int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        மாறிலி int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((எழுத்து *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        மாறிலி src0_t * src0_ptr = (மாறிலி src0_t *) ((மாறிலி எழுத்து *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
நிலையான வெறுமை unary_op_functor(மாறிலி ggml_compute_params * params, ggml_tensor * dst, Op op) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    /*  */ எனில் (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, தசம, தசம>(params, dst, op);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, தசம>(params, dst, op);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, தசம>(params, dst, op);
    } இல்லையெனில் {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

வெறுமை ggml_compute_forward_abs(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

வெறுமை ggml_compute_forward_sgn(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

வெறுமை ggml_compute_forward_neg(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

வெறுமை ggml_compute_forward_step(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

வெறுமை ggml_compute_forward_tanh(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

வெறுமை ggml_compute_forward_elu(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

வெறுமை ggml_compute_forward_relu(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

வெறுமை ggml_compute_forward_sigmoid(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

வெறுமை ggml_compute_forward_hardsigmoid(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

வெறுமை ggml_compute_forward_exp(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

வெறுமை ggml_compute_forward_hardswish(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

வெறுமை ggml_compute_forward_sqr(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

வெறுமை ggml_compute_forward_sqrt(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

வெறுமை ggml_compute_forward_sin(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

வெறுமை ggml_compute_forward_cos(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

வெறுமை ggml_compute_forward_log(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

வெறுமை ggml_compute_forward_expm1(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

வெறுமை ggml_compute_forward_softplus(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

வெறுமை ggml_compute_forward_floor(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

வெறுமை ggml_compute_forward_ceil(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

வெறுமை ggml_compute_forward_round(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

வெறுமை ggml_compute_forward_trunc(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

வெறுமை ggml_compute_forward_xielu(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி தசம alpha_n = ggml_get_op_params_f32(dst, 1);
    மாறிலி தசம alpha_p = ggml_get_op_params_f32(dst, 2);
    மாறிலி தசம beta = ggml_get_op_params_f32(dst, 3);
    மாறிலி தசம eps = ggml_get_op_params_f32(dst, 4);

    மாறிலி auto xielu_op_params = [alpha_n, alpha_p, beta, eps](தசம f) {
        திரும்பு op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

