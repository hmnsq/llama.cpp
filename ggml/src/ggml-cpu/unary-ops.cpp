#समावेश "unary-ops.h"

अटल अंतरभूत भग्न op_abs(भग्न x) {
    वापस fabsf(x);
}

अटल अंतरभूत भग्न op_sgn(भग्न x) {
    वापस (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

अटल अंतरभूत भग्न op_neg(भग्न x) {
    वापस -x;
}

अटल अंतरभूत भग्न op_step(भग्न x) {
    वापस (x > 0.f) ? 1.f : 0.f;
}

अटल अंतरभूत भग्न op_tanh(भग्न x) {
    वापस tanhf(x);
}

अटल अंतरभूत भग्न op_elu(भग्न x) {
    वापस (x > 0.f) ? x : expm1f(x);
}

अटल अंतरभूत भग्न op_relu(भग्न x) {
    वापस (x > 0.f) ? x : 0.f;
}

अटल अंतरभूत भग्न op_sigmoid(भग्न x) {
    वापस 1.f / (1.f + expf(-x));
}

अटल अंतरभूत भग्न op_hardsigmoid(भग्न x) {
    वापस fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

अटल अंतरभूत भग्न op_exp(भग्न x) {
    वापस expf(x);
}

अटल अंतरभूत भग्न op_hardswish(भग्न x) {
    वापस x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

अटल अंतरभूत भग्न op_sqr(भग्न x) {
    वापस x * x;
}

अटल अंतरभूत भग्न op_sqrt(भग्न x) {
    वापस sqrtf(x);
}

अटल अंतरभूत भग्न op_xielu(भग्न x, भग्न alpha_n, भग्न alpha_p, भग्न beta, भग्न eps) {
    अगर (x > 0.0f) {
        वापस alpha_p * x * x + beta * x;
    } अन्यथा {
        स्थिर भग्न min_x_eps = fminf(x, eps);
        वापस (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

अटल अंतरभूत भग्न op_sin(भग्न x) {
    वापस sinf(x);
}

अटल अंतरभूत भग्न op_cos(भग्न x) {
    वापस cosf(x);
}

अटल अंतरभूत भग्न op_log(भग्न x) {
    वापस logf(x);
}

अटल अंतरभूत भग्न op_expm1(भग्न x) {
    वापस expf(x) - 1.0f;
}

अटल अंतरभूत भग्न op_softplus(भग्न x) {
    वापस (x > 20.0f) ? x : logf(1.0f + expf(x));
}

अटल अंतरभूत भग्न op_floor(भग्न x) {
    वापस floorf(x);
}

अटल अंतरभूत भग्न op_ceil(भग्न x) {
    वापस ceilf(x);
}

अटल अंतरभूत भग्न op_round(भग्न x) {
    वापस roundf(x);
}

अटल अंतरभूत भग्न op_trunc(भग्न x) {
    वापस truncf(x);
}

ढाँचा <भग्न (*op)(भग्न), typename src0_t, typename dst_t>
अटल अंतरभूत व्योम vec_unary_op(int64_t n, dst_t * y, स्थिर src0_t * x) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    क्रम (पूर्णांक i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

ढाँचा <भग्न (*op)(भग्न), typename src0_t, typename dst_t>
अटल व्योम apply_unary_op(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == माप(dst_t));
    GGML_ASSERT(nb00 == माप(src0_t));

    स्थिर स्वतः [ir0, ir1] = get_thread_range(params, src0);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne02*ne01);
        स्थिर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((अक्षर *)       dst->तथ्य  + i03*nb3  + i02*nb2  + i01*nb1 );
        स्थिर src0_t * src0_ptr = (स्थिर src0_t *) ((स्थिर अक्षर *) src0->तथ्य + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ढाँचा <भग्न (*op)(भग्न)>
अटल व्योम unary_op(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    /*  */ अगर (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, भग्न, भग्न>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, भग्न>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, भग्न>(params, dst);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ढाँचा <भग्न (*op)(भग्न, ggml_tensor *)>
अटल व्योम unary_op_params(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    /*  */ अगर (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, भग्न, भग्न>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, भग्न>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, भग्न>(params, dst);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
ढाँचा <typename Op, typename src0_t, typename dst_t>
अटल अंतरभूत व्योम vec_unary_op_functor(int64_t n, dst_t * y, स्थिर src0_t * x, Op op) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    क्रम (पूर्णांक i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
ढाँचा <typename Op, typename src0_t, typename dst_t>
अटल व्योम apply_unary_op_functor(स्थिर ggml_compute_params * params, ggml_tensor * dst, Op op) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == माप(dst_t));
    GGML_ASSERT(nb00 == माप(src0_t));

    स्थिर स्वतः [ir0, ir1] = get_thread_range(params, src0);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne02*ne01);
        स्थिर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((अक्षर *)       dst->तथ्य  + i03*nb3  + i02*nb2  + i01*nb1 );
        स्थिर src0_t * src0_ptr = (स्थिर src0_t *) ((स्थिर अक्षर *) src0->तथ्य + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
ढाँचा <typename Op>
अटल व्योम unary_op_functor(स्थिर ggml_compute_params * params, ggml_tensor * dst, Op op) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    /*  */ अगर (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, भग्न, भग्न>(params, dst, op);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, भग्न>(params, dst, op);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, भग्न>(params, dst, op);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

व्योम ggml_compute_forward_abs(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

व्योम ggml_compute_forward_sgn(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

व्योम ggml_compute_forward_neg(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

व्योम ggml_compute_forward_step(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

व्योम ggml_compute_forward_tanh(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

व्योम ggml_compute_forward_elu(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

व्योम ggml_compute_forward_relu(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

व्योम ggml_compute_forward_sigmoid(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

व्योम ggml_compute_forward_hardsigmoid(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

व्योम ggml_compute_forward_exp(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

व्योम ggml_compute_forward_hardswish(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

व्योम ggml_compute_forward_sqr(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

व्योम ggml_compute_forward_sqrt(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

व्योम ggml_compute_forward_sin(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

व्योम ggml_compute_forward_cos(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

व्योम ggml_compute_forward_log(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

व्योम ggml_compute_forward_expm1(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

व्योम ggml_compute_forward_softplus(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

व्योम ggml_compute_forward_floor(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

व्योम ggml_compute_forward_ceil(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

व्योम ggml_compute_forward_round(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

व्योम ggml_compute_forward_trunc(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

व्योम ggml_compute_forward_xielu(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर भग्न alpha_n = ggml_get_op_params_f32(dst, 1);
    स्थिर भग्न alpha_p = ggml_get_op_params_f32(dst, 2);
    स्थिर भग्न beta = ggml_get_op_params_f32(dst, 3);
    स्थिर भग्न eps = ggml_get_op_params_f32(dst, 4);

    स्थिर स्वतः xielu_op_params = [alpha_n, alpha_p, beta, eps](भग्न f) {
        वापस op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

