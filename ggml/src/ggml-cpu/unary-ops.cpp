#شامل "unary-ops.h"

ساکن اندرونی اعشاری op_abs(اعشاری x) {
    واپس fabsf(x);
}

ساکن اندرونی اعشاری op_sgn(اعشاری x) {
    واپس (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

ساکن اندرونی اعشاری op_neg(اعشاری x) {
    واپس -x;
}

ساکن اندرونی اعشاری op_قدم(اعشاری x) {
    واپس (x > 0.f) ? 1.f : 0.f;
}

ساکن اندرونی اعشاری op_tanh(اعشاری x) {
    واپس tanhf(x);
}

ساکن اندرونی اعشاری op_elu(اعشاری x) {
    واپس (x > 0.f) ? x : expm1f(x);
}

ساکن اندرونی اعشاری op_relu(اعشاری x) {
    واپس (x > 0.f) ? x : 0.f;
}

ساکن اندرونی اعشاری op_sigmoid(اعشاری x) {
    واپس 1.f / (1.f + expf(-x));
}

ساکن اندرونی اعشاری op_hardsigmoid(اعشاری x) {
    واپس fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ساکن اندرونی اعشاری op_exp(اعشاری x) {
    واپس expf(x);
}

ساکن اندرونی اعشاری op_hardswish(اعشاری x) {
    واپس x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ساکن اندرونی اعشاری op_sqr(اعشاری x) {
    واپس x * x;
}

ساکن اندرونی اعشاری op_sqrt(اعشاری x) {
    واپس sqrtf(x);
}

ساکن اندرونی اعشاری op_xielu(اعشاری x, اعشاری alpha_n, اعشاری alpha_p, اعشاری beta, اعشاری eps) {
    اگر (x > 0.0f) {
        واپس alpha_p * x * x + beta * x;
    } ورنہ {
        ثابت اعشاری min_x_eps = fminf(x, eps);
        واپس (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

ساکن اندرونی اعشاری op_sin(اعشاری x) {
    واپس sinf(x);
}

ساکن اندرونی اعشاری op_cos(اعشاری x) {
    واپس cosf(x);
}

ساکن اندرونی اعشاری op_log(اعشاری x) {
    واپس logf(x);
}

ساکن اندرونی اعشاری op_expm1(اعشاری x) {
    واپس expf(x) - 1.0f;
}

ساکن اندرونی اعشاری op_softplus(اعشاری x) {
    واپس (x > 20.0f) ? x : logf(1.0f + expf(x));
}

ساکن اندرونی اعشاری op_floor(اعشاری x) {
    واپس floorf(x);
}

ساکن اندرونی اعشاری op_ceil(اعشاری x) {
    واپس ceilf(x);
}

ساکن اندرونی اعشاری op_باری(اعشاری x) {
    واپس roundf(x);
}

ساکن اندرونی اعشاری op_trunc(اعشاری x) {
    واپس truncf(x);
}

سانچہ <اعشاری (*op)(اعشاری), قسم_نام src0_t, قسم_نام dst_t>
ساکن اندرونی خالی vec_unary_op(int64_t n, dst_t * y, ثابت src0_t * x) {
    constexpr خودکار src0_مقصد_f32 = type_conversion_جدول<src0_t>::مقصد_f32;
    constexpr خودکار f32_مقصد_dst  = type_conversion_جدول<dst_t >::from_f32;

    برائے (صحیح i = 0; i < n; i++) {
        y[i] = f32_مقصد_dst(op(src0_مقصد_f32(x[i])));
    }
}

سانچہ <اعشاری (*op)(اعشاری), قسم_نام src0_t, قسم_نام dst_t>
ساکن خالی apply_unary_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_صفیں(src0) && ggml_is_contiguous_صفیں(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(dst_t));
    GGML_ASSERT(nb00 == حجم(src0_t));

    ثابت خودکار [ir0, ir1] = get_thread_range(params, src0);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((حرف *)       dst->ڈیٹا  + i03*nb3  + i02*nb2  + i01*nb1 );
        ثابت src0_t * src0_ptr = (ثابت src0_t *) ((ثابت حرف *) src0->ڈیٹا + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
سانچہ <اعشاری (*op)(اعشاری)>
ساکن خالی unary_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ اگر (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, اعشاری, اعشاری>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, اعشاری>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, اعشاری>(params, dst);
    } ورنہ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

سانچہ <اعشاری (*op)(اعشاری, ggml_tensor *)>
ساکن خالی unary_op_params(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ اگر (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, اعشاری, اعشاری>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, اعشاری>(params, dst);
    } ورنہ اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, اعشاری>(params, dst);
    } ورنہ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
سانچہ <قسم_نام Op, قسم_نام src0_t, قسم_نام dst_t>
ساکن اندرونی خالی vec_unary_op_functor(int64_t n, dst_t * y, ثابت src0_t * x, Op op) {
    constexpr خودکار src0_مقصد_f32 = type_conversion_جدول<src0_t>::مقصد_f32;
    constexpr خودکار f32_مقصد_dst  = type_conversion_جدول<dst_t >::from_f32;

    برائے (صحیح i = 0; i < n; i++) {
        y[i] = f32_مقصد_dst(op(src0_مقصد_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
سانچہ <قسم_نام Op, قسم_نام src0_t, قسم_نام dst_t>
ساکن خالی apply_unary_op_functor(ثابت ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(dst_t));
    GGML_ASSERT(nb00 == حجم(src0_t));

    ثابت خودکار [ir0, ir1] = get_thread_range(params, src0);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((حرف *)       dst->ڈیٹا  + i03*nb3  + i02*nb2  + i01*nb1 );
        ثابت src0_t * src0_ptr = (ثابت src0_t *) ((ثابت حرف *) src0->ڈیٹا + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
سانچہ <قسم_نام Op>
ساکن خالی unary_op_functor(ثابت ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ اگر (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, اعشاری, اعشاری>(params, dst, op);
    } ورنہ اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, اعشاری>(params, dst, op);
    } ورنہ اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, اعشاری>(params, dst, op);
    } ورنہ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

خالی ggml_compute_forward_abs(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

خالی ggml_compute_forward_sgn(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

خالی ggml_compute_forward_neg(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

خالی ggml_compute_forward_قدم(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_قدم>(params, dst);
}

خالی ggml_compute_forward_tanh(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

خالی ggml_compute_forward_elu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

خالی ggml_compute_forward_relu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

خالی ggml_compute_forward_sigmoid(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

خالی ggml_compute_forward_hardsigmoid(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

خالی ggml_compute_forward_exp(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

خالی ggml_compute_forward_hardswish(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

خالی ggml_compute_forward_sqr(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

خالی ggml_compute_forward_sqrt(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

خالی ggml_compute_forward_sin(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

خالی ggml_compute_forward_cos(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

خالی ggml_compute_forward_log(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

خالی ggml_compute_forward_expm1(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

خالی ggml_compute_forward_softplus(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

خالی ggml_compute_forward_floor(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

خالی ggml_compute_forward_ceil(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

خالی ggml_compute_forward_باری(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_باری>(params, dst);
}

خالی ggml_compute_forward_trunc(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

خالی ggml_compute_forward_xielu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت اعشاری alpha_n = ggml_get_op_params_f32(dst, 1);
    ثابت اعشاری alpha_p = ggml_get_op_params_f32(dst, 2);
    ثابت اعشاری beta = ggml_get_op_params_f32(dst, 3);
    ثابت اعشاری eps = ggml_get_op_params_f32(dst, 4);

    ثابت خودکار xielu_op_params = [alpha_n, alpha_p, beta, eps](اعشاری f) {
        واپس op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

