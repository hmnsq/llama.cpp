#অন্তর্ভাৱ "unary-ops.h"

স্থির অন্তঃস্থ ভিন্ন op_abs(ভিন্ন x) {
    ঘূৰাই fabsf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_sgn(ভিন্ন x) {
    ঘূৰাই (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

স্থির অন্তঃস্থ ভিন্ন op_neg(ভিন্ন x) {
    ঘূৰাই -x;
}

স্থির অন্তঃস্থ ভিন্ন op_step(ভিন্ন x) {
    ঘূৰাই (x > 0.f) ? 1.f : 0.f;
}

স্থির অন্তঃস্থ ভিন্ন op_tanh(ভিন্ন x) {
    ঘূৰাই tanhf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_elu(ভিন্ন x) {
    ঘূৰাই (x > 0.f) ? x : expm1f(x);
}

স্থির অন্তঃস্থ ভিন্ন op_relu(ভিন্ন x) {
    ঘূৰাই (x > 0.f) ? x : 0.f;
}

স্থির অন্তঃস্থ ভিন্ন op_sigmoid(ভিন্ন x) {
    ঘূৰাই 1.f / (1.f + expf(-x));
}

স্থির অন্তঃস্থ ভিন্ন op_hardsigmoid(ভিন্ন x) {
    ঘূৰাই fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

স্থির অন্তঃস্থ ভিন্ন op_exp(ভিন্ন x) {
    ঘূৰাই expf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_hardswish(ভিন্ন x) {
    ঘূৰাই x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

স্থির অন্তঃস্থ ভিন্ন op_sqr(ভিন্ন x) {
    ঘূৰাই x * x;
}

স্থির অন্তঃস্থ ভিন্ন op_sqrt(ভিন্ন x) {
    ঘূৰাই sqrtf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_xielu(ভিন্ন x, ভিন্ন alpha_n, ভিন্ন alpha_p, ভিন্ন beta, ভিন্ন eps) {
    যদি (x > 0.0f) {
        ঘূৰাই alpha_p * x * x + beta * x;
    } অন্যথা {
        ধ্রুৱ ভিন্ন min_x_eps = fminf(x, eps);
        ঘূৰাই (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

স্থির অন্তঃস্থ ভিন্ন op_sin(ভিন্ন x) {
    ঘূৰাই sinf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_cos(ভিন্ন x) {
    ঘূৰাই cosf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_log(ভিন্ন x) {
    ঘূৰাই logf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_expm1(ভিন্ন x) {
    ঘূৰাই expf(x) - 1.0f;
}

স্থির অন্তঃস্থ ভিন্ন op_softplus(ভিন্ন x) {
    ঘূৰাই (x > 20.0f) ? x : logf(1.0f + expf(x));
}

স্থির অন্তঃস্থ ভিন্ন op_floor(ভিন্ন x) {
    ঘূৰাই floorf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_ceil(ভিন্ন x) {
    ঘূৰাই ceilf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_round(ভিন্ন x) {
    ঘূৰাই roundf(x);
}

স্থির অন্তঃস্থ ভিন্ন op_trunc(ভিন্ন x) {
    ঘূৰাই truncf(x);
}

template <ভিন্ন (*op)(ভিন্ন), typename src0_t, typename dst_t>
স্থির অন্তঃস্থ শূন্য vec_unary_op(int64_t n, dst_t * y, ধ্রুৱ src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <ভিন্ন (*op)(ভিন্ন), typename src0_t, typename dst_t>
স্থির শূন্য apply_unary_op(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == পরিমাণ(dst_t));
    GGML_ASSERT(nb00 == পরিমাণ(src0_t));

    ধ্রুৱ auto [ir0, ir1] = get_thread_range(params, src0);

    বাবে (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুৱ int64_t i03 = ir/(ne02*ne01);
        ধ্রুৱ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ধ্রুৱ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ৱর্ণ *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ধ্রুৱ src0_t * src0_ptr = (ধ্রুৱ src0_t *) ((ধ্রুৱ ৱর্ণ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ভিন্ন (*op)(ভিন্ন)>
স্থির শূন্য unary_op(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];

    /*  */ যদি (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ভিন্ন, ভিন্ন>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ভিন্ন>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ভিন্ন>(params, dst);
    } অন্যথা {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <ভিন্ন (*op)(ভিন্ন, ggml_tensor *)>
স্থির শূন্য unary_op_params(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];

    /*  */ যদি (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ভিন্ন, ভিন্ন>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ভিন্ন>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ভিন্ন>(params, dst);
    } অন্যথা {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
স্থির অন্তঃস্থ শূন্য vec_unary_op_functor(int64_t n, dst_t * y, ধ্রুৱ src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
স্থির শূন্য apply_unary_op_functor(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == পরিমাণ(dst_t));
    GGML_ASSERT(nb00 == পরিমাণ(src0_t));

    ধ্রুৱ auto [ir0, ir1] = get_thread_range(params, src0);

    বাবে (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুৱ int64_t i03 = ir/(ne02*ne01);
        ধ্রুৱ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ধ্রুৱ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((ৱর্ণ *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ধ্রুৱ src0_t * src0_ptr = (ধ্রুৱ src0_t *) ((ধ্রুৱ ৱর্ণ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
স্থির শূন্য unary_op_functor(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];

    /*  */ যদি (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, ভিন্ন, ভিন্ন>(params, dst, op);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, ভিন্ন>(params, dst, op);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, ভিন্ন>(params, dst, op);
    } অন্যথা {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

শূন্য ggml_compute_forward_abs(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

শূন্য ggml_compute_forward_sgn(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

শূন্য ggml_compute_forward_neg(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

শূন্য ggml_compute_forward_step(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

শূন্য ggml_compute_forward_tanh(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

শূন্য ggml_compute_forward_elu(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

শূন্য ggml_compute_forward_relu(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

শূন্য ggml_compute_forward_sigmoid(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

শূন্য ggml_compute_forward_hardsigmoid(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

শূন্য ggml_compute_forward_exp(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

শূন্য ggml_compute_forward_hardswish(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

শূন্য ggml_compute_forward_sqr(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

শূন্য ggml_compute_forward_sqrt(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

শূন্য ggml_compute_forward_sin(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

শূন্য ggml_compute_forward_cos(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

শূন্য ggml_compute_forward_log(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

শূন্য ggml_compute_forward_expm1(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

শূন্য ggml_compute_forward_softplus(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

শূন্য ggml_compute_forward_floor(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

শূন্য ggml_compute_forward_ceil(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

শূন্য ggml_compute_forward_round(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

শূন্য ggml_compute_forward_trunc(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

শূন্য ggml_compute_forward_xielu(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুৱ ভিন্ন alpha_n = ggml_get_op_params_f32(dst, 1);
    ধ্রুৱ ভিন্ন alpha_p = ggml_get_op_params_f32(dst, 2);
    ধ্রুৱ ভিন্ন beta = ggml_get_op_params_f32(dst, 3);
    ধ্রুৱ ভিন্ন eps = ggml_get_op_params_f32(dst, 4);

    ধ্রুৱ auto xielu_op_params = [alpha_n, alpha_p, beta, eps](ভিন্ন f) {
        ঘূৰাই op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

