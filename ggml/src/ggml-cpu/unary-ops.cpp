#अन्तर्भाव "unary-ops.h"

स्थिर अन्तःस्थ भिन्न op_abs(भिन्न x) {
    फर्काऊ fabsf(x);
}

स्थिर अन्तःस्थ भिन्न op_sgn(भिन्न x) {
    फर्काऊ (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

स्थिर अन्तःस्थ भिन्न op_neg(भिन्न x) {
    फर्काऊ -x;
}

स्थिर अन्तःस्थ भिन्न op_step(भिन्न x) {
    फर्काऊ (x > 0.f) ? 1.f : 0.f;
}

स्थिर अन्तःस्थ भिन्न op_tanh(भिन्न x) {
    फर्काऊ tanhf(x);
}

स्थिर अन्तःस्थ भिन्न op_elu(भिन्न x) {
    फर्काऊ (x > 0.f) ? x : expm1f(x);
}

स्थिर अन्तःस्थ भिन्न op_relu(भिन्न x) {
    फर्काऊ (x > 0.f) ? x : 0.f;
}

स्थिर अन्तःस्थ भिन्न op_sigmoid(भिन्न x) {
    फर्काऊ 1.f / (1.f + expf(-x));
}

स्थिर अन्तःस्थ भिन्न op_hardsigmoid(भिन्न x) {
    फर्काऊ fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

स्थिर अन्तःस्थ भिन्न op_exp(भिन्न x) {
    फर्काऊ expf(x);
}

स्थिर अन्तःस्थ भिन्न op_hardswish(भिन्न x) {
    फर्काऊ x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

स्थिर अन्तःस्थ भिन्न op_sqr(भिन्न x) {
    फर्काऊ x * x;
}

स्थिर अन्तःस्थ भिन्न op_sqrt(भिन्न x) {
    फर्काऊ sqrtf(x);
}

स्थिर अन्तःस्थ भिन्न op_xielu(भिन्न x, भिन्न alpha_n, भिन्न alpha_p, भिन्न beta, भिन्न eps) {
    यदि (x > 0.0f) {
        फर्काऊ alpha_p * x * x + beta * x;
    } अन्यथा {
        अचर भिन्न min_x_eps = fminf(x, eps);
        फर्काऊ (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

स्थिर अन्तःस्थ भिन्न op_sin(भिन्न x) {
    फर्काऊ sinf(x);
}

स्थिर अन्तःस्थ भिन्न op_cos(भिन्न x) {
    फर्काऊ cosf(x);
}

स्थिर अन्तःस्थ भिन्न op_log(भिन्न x) {
    फर्काऊ logf(x);
}

स्थिर अन्तःस्थ भिन्न op_expm1(भिन्न x) {
    फर्काऊ expf(x) - 1.0f;
}

स्थिर अन्तःस्थ भिन्न op_softplus(भिन्न x) {
    फर्काऊ (x > 20.0f) ? x : logf(1.0f + expf(x));
}

स्थिर अन्तःस्थ भिन्न op_floor(भिन्न x) {
    फर्काऊ floorf(x);
}

स्थिर अन्तःस्थ भिन्न op_ceil(भिन्न x) {
    फर्काऊ ceilf(x);
}

स्थिर अन्तःस्थ भिन्न op_round(भिन्न x) {
    फर्काऊ roundf(x);
}

स्थिर अन्तःस्थ भिन्न op_trunc(भिन्न x) {
    फर्काऊ truncf(x);
}

ढाँचा <भिन्न (*op)(भिन्न), typename src0_t, typename dst_t>
स्थिर अन्तःस्थ शून्य vec_unary_op(int64_t n, dst_t * y, अचर src0_t * x) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

ढाँचा <भिन्न (*op)(भिन्न), typename src0_t, typename dst_t>
स्थिर शून्य apply_unary_op(अचर ggml_compute_params * params, ggml_tensor * dst) {
    अचर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(dst_t));
    GGML_ASSERT(nb00 == परिमाण(src0_t));

    अचर स्वतः [ir0, ir1] = get_thread_range(params, src0);

    लागि (int64_t ir = ir0; ir < ir1; ++ir) {
        अचर int64_t i03 = ir/(ne02*ne01);
        अचर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        अचर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((वर्ण *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        अचर src0_t * src0_ptr = (अचर src0_t *) ((अचर वर्ण *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ढाँचा <भिन्न (*op)(भिन्न)>
स्थिर शून्य unary_op(अचर ggml_compute_params * params, ggml_tensor * dst) {
    अचर ggml_tensor * src0 = dst->src[0];

    /*  */ यदि (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, भिन्न, भिन्न>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, भिन्न>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, भिन्न>(params, dst);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ढाँचा <भिन्न (*op)(भिन्न, ggml_tensor *)>
स्थिर शून्य unary_op_params(अचर ggml_compute_params * params, ggml_tensor * dst) {
    अचर ggml_tensor * src0 = dst->src[0];

    /*  */ यदि (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, भिन्न, भिन्न>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, भिन्न>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, भिन्न>(params, dst);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
ढाँचा <typename Op, typename src0_t, typename dst_t>
स्थिर अन्तःस्थ शून्य vec_unary_op_functor(int64_t n, dst_t * y, अचर src0_t * x, Op op) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
ढाँचा <typename Op, typename src0_t, typename dst_t>
स्थिर शून्य apply_unary_op_functor(अचर ggml_compute_params * params, ggml_tensor * dst, Op op) {
    अचर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(dst_t));
    GGML_ASSERT(nb00 == परिमाण(src0_t));

    अचर स्वतः [ir0, ir1] = get_thread_range(params, src0);

    लागि (int64_t ir = ir0; ir < ir1; ++ir) {
        अचर int64_t i03 = ir/(ne02*ne01);
        अचर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        अचर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((वर्ण *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        अचर src0_t * src0_ptr = (अचर src0_t *) ((अचर वर्ण *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
ढाँचा <typename Op>
स्थिर शून्य unary_op_functor(अचर ggml_compute_params * params, ggml_tensor * dst, Op op) {
    अचर ggml_tensor * src0 = dst->src[0];

    /*  */ यदि (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, भिन्न, भिन्न>(params, dst, op);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, भिन्न>(params, dst, op);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, भिन्न>(params, dst, op);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

शून्य ggml_compute_forward_abs(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

शून्य ggml_compute_forward_sgn(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

शून्य ggml_compute_forward_neg(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

शून्य ggml_compute_forward_step(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

शून्य ggml_compute_forward_tanh(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

शून्य ggml_compute_forward_elu(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

शून्य ggml_compute_forward_relu(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

शून्य ggml_compute_forward_sigmoid(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

शून्य ggml_compute_forward_hardsigmoid(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

शून्य ggml_compute_forward_exp(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

शून्य ggml_compute_forward_hardswish(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

शून्य ggml_compute_forward_sqr(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

शून्य ggml_compute_forward_sqrt(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

शून्य ggml_compute_forward_sin(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

शून्य ggml_compute_forward_cos(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

शून्य ggml_compute_forward_log(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

शून्य ggml_compute_forward_expm1(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

शून्य ggml_compute_forward_softplus(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

शून्य ggml_compute_forward_floor(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

शून्य ggml_compute_forward_ceil(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

शून्य ggml_compute_forward_round(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

शून्य ggml_compute_forward_trunc(अचर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

शून्य ggml_compute_forward_xielu(अचर ggml_compute_params * params, ggml_tensor * dst) {
    अचर भिन्न alpha_n = ggml_get_op_params_f32(dst, 1);
    अचर भिन्न alpha_p = ggml_get_op_params_f32(dst, 2);
    अचर भिन्न beta = ggml_get_op_params_f32(dst, 3);
    अचर भिन्न eps = ggml_get_op_params_f32(dst, 4);

    अचर स्वतः xielu_op_params = [alpha_n, alpha_p, beta, eps](भिन्न f) {
        फर्काऊ op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

