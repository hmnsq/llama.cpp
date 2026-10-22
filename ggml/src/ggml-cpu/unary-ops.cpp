#ضمن "unary-ops.h"

ساكن مضمن عائم op_abs(عائم x) {
    أرجع fabsf(x);
}

ساكن مضمن عائم op_sgn(عائم x) {
    أرجع (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

ساكن مضمن عائم op_neg(عائم x) {
    أرجع -x;
}

ساكن مضمن عائم op_قدم(عائم x) {
    أرجع (x > 0.f) ? 1.f : 0.f;
}

ساكن مضمن عائم op_tanh(عائم x) {
    أرجع tanhf(x);
}

ساكن مضمن عائم op_elu(عائم x) {
    أرجع (x > 0.f) ? x : expm1f(x);
}

ساكن مضمن عائم op_relu(عائم x) {
    أرجع (x > 0.f) ? x : 0.f;
}

ساكن مضمن عائم op_sigmoid(عائم x) {
    أرجع 1.f / (1.f + expf(-x));
}

ساكن مضمن عائم op_hardsigmoid(عائم x) {
    أرجع fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ساكن مضمن عائم op_exp(عائم x) {
    أرجع expf(x);
}

ساكن مضمن عائم op_hardswish(عائم x) {
    أرجع x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ساكن مضمن عائم op_sqr(عائم x) {
    أرجع x * x;
}

ساكن مضمن عائم op_sqrt(عائم x) {
    أرجع sqrtf(x);
}

ساكن مضمن عائم op_xielu(عائم x, عائم alpha_n, عائم alpha_p, عائم beta, عائم eps) {
    إذا (x > 0.0f) {
        أرجع alpha_p * x * x + beta * x;
    } وإلا {
        ثابت عائم min_x_eps = fminf(x, eps);
        أرجع (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

ساكن مضمن عائم op_sin(عائم x) {
    أرجع sinf(x);
}

ساكن مضمن عائم op_cos(عائم x) {
    أرجع cosf(x);
}

ساكن مضمن عائم op_log(عائم x) {
    أرجع logf(x);
}

ساكن مضمن عائم op_expm1(عائم x) {
    أرجع expf(x) - 1.0f;
}

ساكن مضمن عائم op_softplus(عائم x) {
    أرجع (x > 20.0f) ? x : logf(1.0f + expf(x));
}

ساكن مضمن عائم op_floor(عائم x) {
    أرجع floorf(x);
}

ساكن مضمن عائم op_ceil(عائم x) {
    أرجع ceilf(x);
}

ساكن مضمن عائم op_ڦيرو(عائم x) {
    أرجع roundf(x);
}

ساكن مضمن عائم op_trunc(عائم x) {
    أرجع truncf(x);
}

سانچو <عائم (*op)(عائم), قسم_نالو src0_t, قسم_نالو dst_t>
ساكن مضمن فراغ vec_unary_op(int64_t n, dst_t * y, ثابت src0_t * x) {
    constexpr خودڪار src0_منزل_f32 = type_conversion_جدول<src0_t>::منزل_f32;
    constexpr خودڪار f32_منزل_dst  = type_conversion_جدول<dst_t >::from_f32;

    لكل (صحيح i = 0; i < n; i++) {
        y[i] = f32_منزل_dst(op(src0_منزل_f32(x[i])));
    }
}

سانچو <عائم (*op)(عائم), قسم_نالو src0_t, قسم_نالو dst_t>
ساكن فراغ apply_unary_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_صفون(src0) && ggml_is_contiguous_صفون(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(dst_t));
    GGML_ASSERT(nb00 == حجم(src0_t));

    ثابت خودڪار [ir0, ir1] = get_thread_range(params, src0);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((حرف *)       dst->معلومات  + i03*nb3  + i02*nb2  + i01*nb1 );
        ثابت src0_t * src0_ptr = (ثابت src0_t *) ((ثابت حرف *) src0->معلومات + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
سانچو <عائم (*op)(عائم)>
ساكن فراغ unary_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ إذا (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, عائم, عائم>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, عائم>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, عائم>(params, dst);
    } وإلا {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

سانچو <عائم (*op)(عائم, ggml_tensor *)>
ساكن فراغ unary_op_params(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ إذا (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, عائم, عائم>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, عائم>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, عائم>(params, dst);
    } وإلا {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
سانچو <قسم_نالو Op, قسم_نالو src0_t, قسم_نالو dst_t>
ساكن مضمن فراغ vec_unary_op_functor(int64_t n, dst_t * y, ثابت src0_t * x, Op op) {
    constexpr خودڪار src0_منزل_f32 = type_conversion_جدول<src0_t>::منزل_f32;
    constexpr خودڪار f32_منزل_dst  = type_conversion_جدول<dst_t >::from_f32;

    لكل (صحيح i = 0; i < n; i++) {
        y[i] = f32_منزل_dst(op(src0_منزل_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
سانچو <قسم_نالو Op, قسم_نالو src0_t, قسم_نالو dst_t>
ساكن فراغ apply_unary_op_functor(ثابت ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(dst_t));
    GGML_ASSERT(nb00 == حجم(src0_t));

    ثابت خودڪار [ir0, ir1] = get_thread_range(params, src0);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((حرف *)       dst->معلومات  + i03*nb3  + i02*nb2  + i01*nb1 );
        ثابت src0_t * src0_ptr = (ثابت src0_t *) ((ثابت حرف *) src0->معلومات + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
سانچو <قسم_نالو Op>
ساكن فراغ unary_op_functor(ثابت ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ إذا (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, عائم, عائم>(params, dst, op);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, عائم>(params, dst, op);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, عائم>(params, dst, op);
    } وإلا {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

فراغ ggml_compute_forward_abs(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

فراغ ggml_compute_forward_sgn(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

فراغ ggml_compute_forward_neg(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

فراغ ggml_compute_forward_قدم(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_قدم>(params, dst);
}

فراغ ggml_compute_forward_tanh(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

فراغ ggml_compute_forward_elu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

فراغ ggml_compute_forward_relu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

فراغ ggml_compute_forward_sigmoid(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

فراغ ggml_compute_forward_hardsigmoid(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

فراغ ggml_compute_forward_exp(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

فراغ ggml_compute_forward_hardswish(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

فراغ ggml_compute_forward_sqr(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

فراغ ggml_compute_forward_sqrt(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

فراغ ggml_compute_forward_sin(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

فراغ ggml_compute_forward_cos(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

فراغ ggml_compute_forward_log(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

فراغ ggml_compute_forward_expm1(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

فراغ ggml_compute_forward_softplus(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

فراغ ggml_compute_forward_floor(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

فراغ ggml_compute_forward_ceil(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

فراغ ggml_compute_forward_ڦيرو(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ڦيرو>(params, dst);
}

فراغ ggml_compute_forward_trunc(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

فراغ ggml_compute_forward_xielu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت عائم alpha_n = ggml_get_op_params_f32(dst, 1);
    ثابت عائم alpha_p = ggml_get_op_params_f32(dst, 2);
    ثابت عائم beta = ggml_get_op_params_f32(dst, 3);
    ثابت عائم eps = ggml_get_op_params_f32(dst, 4);

    ثابت خودڪار xielu_op_params = [alpha_n, alpha_p, beta, eps](عائم f) {
        أرجع op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

