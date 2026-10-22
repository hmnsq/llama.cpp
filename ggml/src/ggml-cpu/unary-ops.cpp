#شامل "unary-ops.h"

ایستا درونخطی اعشاری op_abs(اعشاری x) {
    بازگشت fabsf(x);
}

ایستا درونخطی اعشاری op_sgn(اعشاری x) {
    بازگشت (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

ایستا درونخطی اعشاری op_neg(اعشاری x) {
    بازگشت -x;
}

ایستا درونخطی اعشاری op_step(اعشاری x) {
    بازگشت (x > 0.f) ? 1.f : 0.f;
}

ایستا درونخطی اعشاری op_tanh(اعشاری x) {
    بازگشت tanhf(x);
}

ایستا درونخطی اعشاری op_elu(اعشاری x) {
    بازگشت (x > 0.f) ? x : expm1f(x);
}

ایستا درونخطی اعشاری op_relu(اعشاری x) {
    بازگشت (x > 0.f) ? x : 0.f;
}

ایستا درونخطی اعشاری op_sigmoid(اعشاری x) {
    بازگشت 1.f / (1.f + expf(-x));
}

ایستا درونخطی اعشاری op_hardsigmoid(اعشاری x) {
    بازگشت fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ایستا درونخطی اعشاری op_exp(اعشاری x) {
    بازگشت expf(x);
}

ایستا درونخطی اعشاری op_hardswish(اعشاری x) {
    بازگشت x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

ایستا درونخطی اعشاری op_sqr(اعشاری x) {
    بازگشت x * x;
}

ایستا درونخطی اعشاری op_sqrt(اعشاری x) {
    بازگشت sqrtf(x);
}

ایستا درونخطی اعشاری op_xielu(اعشاری x, اعشاری alpha_n, اعشاری alpha_p, اعشاری beta, اعشاری eps) {
    اگر (x > 0.0f) {
        بازگشت alpha_p * x * x + beta * x;
    } وگرنه {
        ثابت اعشاری min_x_eps = fminf(x, eps);
        بازگشت (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

ایستا درونخطی اعشاری op_sin(اعشاری x) {
    بازگشت sinf(x);
}

ایستا درونخطی اعشاری op_cos(اعشاری x) {
    بازگشت cosf(x);
}

ایستا درونخطی اعشاری op_log(اعشاری x) {
    بازگشت logf(x);
}

ایستا درونخطی اعشاری op_expm1(اعشاری x) {
    بازگشت expf(x) - 1.0f;
}

ایستا درونخطی اعشاری op_softplus(اعشاری x) {
    بازگشت (x > 20.0f) ? x : logf(1.0f + expf(x));
}

ایستا درونخطی اعشاری op_floor(اعشاری x) {
    بازگشت floorf(x);
}

ایستا درونخطی اعشاری op_ceil(اعشاری x) {
    بازگشت ceilf(x);
}

ایستا درونخطی اعشاری op_round(اعشاری x) {
    بازگشت roundf(x);
}

ایستا درونخطی اعشاری op_trunc(اعشاری x) {
    بازگشت truncf(x);
}

template <اعشاری (*op)(اعشاری), typename src0_t, typename dst_t>
ایستا درونخطی تهی vec_unary_op(int64_t n, dst_t * y, ثابت src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    برای (صحیح i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <اعشاری (*op)(اعشاری), typename src0_t, typename dst_t>
ایستا تهی apply_unary_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == اندازه(dst_t));
    GGML_ASSERT(nb00 == اندازه(src0_t));

    ثابت auto [ir0, ir1] = get_thread_range(params, src0);

    برای (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((نویسه *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ثابت src0_t * src0_ptr = (ثابت src0_t *) ((ثابت نویسه *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <اعشاری (*op)(اعشاری)>
ایستا تهی unary_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ اگر (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, اعشاری, اعشاری>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, اعشاری>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, اعشاری>(params, dst);
    } وگرنه {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <اعشاری (*op)(اعشاری, ggml_tensor *)>
ایستا تهی unary_op_params(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ اگر (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, اعشاری, اعشاری>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, اعشاری>(params, dst);
    } وگرنه اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, اعشاری>(params, dst);
    } وگرنه {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
ایستا درونخطی تهی vec_unary_op_functor(int64_t n, dst_t * y, ثابت src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    برای (صحیح i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
ایستا تهی apply_unary_op_functor(ثابت ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == اندازه(dst_t));
    GGML_ASSERT(nb00 == اندازه(src0_t));

    ثابت auto [ir0, ir1] = get_thread_range(params, src0);

    برای (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((نویسه *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ثابت src0_t * src0_ptr = (ثابت src0_t *) ((ثابت نویسه *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
ایستا تهی unary_op_functor(ثابت ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ثابت ggml_tensor * src0 = dst->src[0];

    /*  */ اگر (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, اعشاری, اعشاری>(params, dst, op);
    } وگرنه اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } وگرنه اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } وگرنه اگر (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, اعشاری>(params, dst, op);
    } وگرنه اگر (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, اعشاری>(params, dst, op);
    } وگرنه {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

تهی ggml_compute_forward_abs(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

تهی ggml_compute_forward_sgn(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

تهی ggml_compute_forward_neg(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

تهی ggml_compute_forward_step(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

تهی ggml_compute_forward_tanh(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

تهی ggml_compute_forward_elu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

تهی ggml_compute_forward_relu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

تهی ggml_compute_forward_sigmoid(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

تهی ggml_compute_forward_hardsigmoid(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

تهی ggml_compute_forward_exp(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

تهی ggml_compute_forward_hardswish(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

تهی ggml_compute_forward_sqr(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

تهی ggml_compute_forward_sqrt(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

تهی ggml_compute_forward_sin(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

تهی ggml_compute_forward_cos(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

تهی ggml_compute_forward_log(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

تهی ggml_compute_forward_expm1(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

تهی ggml_compute_forward_softplus(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

تهی ggml_compute_forward_floor(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

تهی ggml_compute_forward_ceil(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

تهی ggml_compute_forward_round(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

تهی ggml_compute_forward_trunc(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

تهی ggml_compute_forward_xielu(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت اعشاری alpha_n = ggml_get_op_params_f32(dst, 1);
    ثابت اعشاری alpha_p = ggml_get_op_params_f32(dst, 2);
    ثابت اعشاری beta = ggml_get_op_params_f32(dst, 3);
    ثابت اعشاری eps = ggml_get_op_params_f32(dst, 4);

    ثابت auto xielu_op_params = [alpha_n, alpha_p, beta, eps](اعشاری f) {
        بازگشت op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

