#ଅନ୍ତର୍ଭାଵ "unary-ops.h"

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_abs(ଦଶମିକ x) {
    ଫେରନ୍ତୁ fabsf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_sgn(ଦଶମିକ x) {
    ଫେରନ୍ତୁ (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_neg(ଦଶମିକ x) {
    ଫେରନ୍ତୁ -x;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_step(ଦଶମିକ x) {
    ଫେରନ୍ତୁ (x > 0.f) ? 1.f : 0.f;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_tanh(ଦଶମିକ x) {
    ଫେରନ୍ତୁ tanhf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_elu(ଦଶମିକ x) {
    ଫେରନ୍ତୁ (x > 0.f) ? x : expm1f(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_relu(ଦଶମିକ x) {
    ଫେରନ୍ତୁ (x > 0.f) ? x : 0.f;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_sigmoid(ଦଶମିକ x) {
    ଫେରନ୍ତୁ 1.f / (1.f + expf(-x));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_hardsigmoid(ଦଶମିକ x) {
    ଫେରନ୍ତୁ fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_exp(ଦଶମିକ x) {
    ଫେରନ୍ତୁ expf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_hardswish(ଦଶମିକ x) {
    ଫେରନ୍ତୁ x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_sqr(ଦଶମିକ x) {
    ଫେରନ୍ତୁ x * x;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_sqrt(ଦଶମିକ x) {
    ଫେରନ୍ତୁ sqrtf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_xielu(ଦଶମିକ x, ଦଶମିକ alpha_n, ଦଶମିକ alpha_p, ଦଶମିକ beta, ଦଶମିକ eps) {
    ଯଦି (x > 0.0f) {
        ଫେରନ୍ତୁ alpha_p * x * x + beta * x;
    } ଅନ୍ୟଥା {
        ଅଚଳ ଦଶମିକ min_x_eps = fminf(x, eps);
        ଫେରନ୍ତୁ (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_sin(ଦଶମିକ x) {
    ଫେରନ୍ତୁ sinf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_cos(ଦଶମିକ x) {
    ଫେରନ୍ତୁ cosf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_log(ଦଶମିକ x) {
    ଫେରନ୍ତୁ logf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_expm1(ଦଶମିକ x) {
    ଫେରନ୍ତୁ expf(x) - 1.0f;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_softplus(ଦଶମିକ x) {
    ଫେରନ୍ତୁ (x > 20.0f) ? x : logf(1.0f + expf(x));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_floor(ଦଶମିକ x) {
    ଫେରନ୍ତୁ floorf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_ceil(ଦଶମିକ x) {
    ଫେରନ୍ତୁ ceilf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_round(ଦଶମିକ x) {
    ଫେରନ୍ତୁ roundf(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_trunc(ଦଶମିକ x) {
    ଫେରନ୍ତୁ truncf(x);
}

template <ଦଶମିକ (*op)(ଦଶମିକ), typename src0_t, typename dst_t>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ vec_unary_op(int64_t n, dst_t * y, ଅଚଳ src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <ଦଶମିକ (*op)(ଦଶମିକ), typename src0_t, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ apply_unary_op(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ଆକାର(dst_t));
    GGML_ASSERT(nb00 == ଆକାର(src0_t));

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, src0);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne02*ne01);
        ଅଚଳ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ଅଚଳ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ଅକ୍ଷର *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ଅଚଳ src0_t * src0_ptr = (ଅଚଳ src0_t *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ଦଶମିକ (*op)(ଦଶମିକ)>
ସ୍ଥିର ଶୂନ୍ୟ unary_op(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    /*  */ ଯଦି (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ଦଶମିକ, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <ଦଶମିକ (*op)(ଦଶମିକ, ggml_tensor *)>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_params(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    /*  */ ଯଦି (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ଦଶମିକ, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ vec_unary_op_functor(int64_t n, dst_t * y, ଅଚଳ src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ apply_unary_op_functor(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ଆକାର(dst_t));
    GGML_ASSERT(nb00 == ଆକାର(src0_t));

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, src0);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne02*ne01);
        ଅଚଳ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ଅଚଳ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ଅକ୍ଷର *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ଅଚଳ src0_t * src0_ptr = (ଅଚଳ src0_t *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_functor(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    /*  */ ଯଦି (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, ଦଶମିକ, ଦଶମିକ>(params, dst, op);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, ଦଶମିକ>(params, dst, op);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, ଦଶମିକ>(params, dst, op);
    } ଅନ୍ୟଥା {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ଶୂନ୍ୟ ggml_compute_forward_abs(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_sgn(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_neg(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_step(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_tanh(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_elu(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_relu(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_sigmoid(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_hardsigmoid(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_exp(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_hardswish(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_sqr(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_sqrt(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_sin(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_cos(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_log(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_expm1(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_softplus(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_floor(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_ceil(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_round(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_trunc(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_xielu(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ଦଶମିକ alpha_n = ggml_get_op_params_f32(dst, 1);
    ଅଚଳ ଦଶମିକ alpha_p = ggml_get_op_params_f32(dst, 2);
    ଅଚଳ ଦଶମିକ beta = ggml_get_op_params_f32(dst, 3);
    ଅଚଳ ଦଶମିକ eps = ggml_get_op_params_f32(dst, 4);

    ଅଚଳ auto xielu_op_params = [alpha_n, alpha_p, beta, eps](ଦଶମିକ f) {
        ଫେରନ୍ତୁ op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

