#अन्तर्भाव "unary-ops.h"

स्थिर ओळीत दशांश op_abs(दशांश x) {
    परत fabsf(x);
}

स्थिर ओळीत दशांश op_sgn(दशांश x) {
    परत (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

स्थिर ओळीत दशांश op_neg(दशांश x) {
    परत -x;
}

स्थिर ओळीत दशांश op_step(दशांश x) {
    परत (x > 0.f) ? 1.f : 0.f;
}

स्थिर ओळीत दशांश op_tanh(दशांश x) {
    परत tanhf(x);
}

स्थिर ओळीत दशांश op_elu(दशांश x) {
    परत (x > 0.f) ? x : expm1f(x);
}

स्थिर ओळीत दशांश op_relu(दशांश x) {
    परत (x > 0.f) ? x : 0.f;
}

स्थिर ओळीत दशांश op_sigmoid(दशांश x) {
    परत 1.f / (1.f + expf(-x));
}

स्थिर ओळीत दशांश op_hardsigmoid(दशांश x) {
    परत fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

स्थिर ओळीत दशांश op_exp(दशांश x) {
    परत expf(x);
}

स्थिर ओळीत दशांश op_hardswish(दशांश x) {
    परत x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

स्थिर ओळीत दशांश op_sqr(दशांश x) {
    परत x * x;
}

स्थिर ओळीत दशांश op_sqrt(दशांश x) {
    परत sqrtf(x);
}

स्थिर ओळीत दशांश op_xielu(दशांश x, दशांश alpha_n, दशांश alpha_p, दशांश beta, दशांश eps) {
    जर (x > 0.0f) {
        परत alpha_p * x * x + beta * x;
    } अन्यथा {
        स्थिरांक दशांश min_x_eps = fminf(x, eps);
        परत (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

स्थिर ओळीत दशांश op_sin(दशांश x) {
    परत sinf(x);
}

स्थिर ओळीत दशांश op_cos(दशांश x) {
    परत cosf(x);
}

स्थिर ओळीत दशांश op_log(दशांश x) {
    परत logf(x);
}

स्थिर ओळीत दशांश op_expm1(दशांश x) {
    परत expf(x) - 1.0f;
}

स्थिर ओळीत दशांश op_softplus(दशांश x) {
    परत (x > 20.0f) ? x : logf(1.0f + expf(x));
}

स्थिर ओळीत दशांश op_floor(दशांश x) {
    परत floorf(x);
}

स्थिर ओळीत दशांश op_ceil(दशांश x) {
    परत ceilf(x);
}

स्थिर ओळीत दशांश op_round(दशांश x) {
    परत roundf(x);
}

स्थिर ओळीत दशांश op_trunc(दशांश x) {
    परत truncf(x);
}

ढाँचा <दशांश (*op)(दशांश), typename src0_t, typename dst_t>
स्थिर ओळीत शून्य vec_unary_op(int64_t n, dst_t * y, स्थिरांक src0_t * x) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    साठी (पूर्णांक i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

ढाँचा <दशांश (*op)(दशांश), typename src0_t, typename dst_t>
स्थिर शून्य apply_unary_op(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == आकार(dst_t));
    GGML_ASSERT(nb00 == आकार(src0_t));

    स्थिरांक स्वतः [ir0, ir1] = get_thread_range(params, src0);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne02*ne01);
        स्थिरांक int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिरांक int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((अक्षर *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        स्थिरांक src0_t * src0_ptr = (स्थिरांक src0_t *) ((स्थिरांक अक्षर *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ढाँचा <दशांश (*op)(दशांश)>
स्थिर शून्य unary_op(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    /*  */ जर (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, दशांश, दशांश>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, दशांश>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, दशांश>(params, dst);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ढाँचा <दशांश (*op)(दशांश, ggml_tensor *)>
स्थिर शून्य unary_op_params(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    /*  */ जर (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, दशांश, दशांश>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, दशांश>(params, dst);
    } अन्यथा जर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, दशांश>(params, dst);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
ढाँचा <typename Op, typename src0_t, typename dst_t>
स्थिर ओळीत शून्य vec_unary_op_functor(int64_t n, dst_t * y, स्थिरांक src0_t * x, Op op) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    साठी (पूर्णांक i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
ढाँचा <typename Op, typename src0_t, typename dst_t>
स्थिर शून्य apply_unary_op_functor(स्थिरांक ggml_compute_params * params, ggml_tensor * dst, Op op) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == आकार(dst_t));
    GGML_ASSERT(nb00 == आकार(src0_t));

    स्थिरांक स्वतः [ir0, ir1] = get_thread_range(params, src0);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne02*ne01);
        स्थिरांक int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिरांक int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((अक्षर *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        स्थिरांक src0_t * src0_ptr = (स्थिरांक src0_t *) ((स्थिरांक अक्षर *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
ढाँचा <typename Op>
स्थिर शून्य unary_op_functor(स्थिरांक ggml_compute_params * params, ggml_tensor * dst, Op op) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    /*  */ जर (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, दशांश, दशांश>(params, dst, op);
    } अन्यथा जर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, दशांश>(params, dst, op);
    } अन्यथा जर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, दशांश>(params, dst, op);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

शून्य ggml_compute_forward_abs(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

शून्य ggml_compute_forward_sgn(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

शून्य ggml_compute_forward_neg(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

शून्य ggml_compute_forward_step(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

शून्य ggml_compute_forward_tanh(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

शून्य ggml_compute_forward_elu(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

शून्य ggml_compute_forward_relu(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

शून्य ggml_compute_forward_sigmoid(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

शून्य ggml_compute_forward_hardsigmoid(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

शून्य ggml_compute_forward_exp(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

शून्य ggml_compute_forward_hardswish(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

शून्य ggml_compute_forward_sqr(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

शून्य ggml_compute_forward_sqrt(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

शून्य ggml_compute_forward_sin(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

शून्य ggml_compute_forward_cos(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

शून्य ggml_compute_forward_log(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

शून्य ggml_compute_forward_expm1(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

शून्य ggml_compute_forward_softplus(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

शून्य ggml_compute_forward_floor(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

शून्य ggml_compute_forward_ceil(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

शून्य ggml_compute_forward_round(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

शून्य ggml_compute_forward_trunc(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

शून्य ggml_compute_forward_xielu(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक दशांश alpha_n = ggml_get_op_params_f32(dst, 1);
    स्थिरांक दशांश alpha_p = ggml_get_op_params_f32(dst, 2);
    स्थिरांक दशांश beta = ggml_get_op_params_f32(dst, 3);
    स्थिरांक दशांश eps = ggml_get_op_params_f32(dst, 4);

    स्थिरांक स्वतः xielu_op_params = [alpha_n, alpha_p, beta, eps](दशांश f) {
        परत op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

