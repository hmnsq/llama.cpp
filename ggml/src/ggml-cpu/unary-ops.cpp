#അന്തര്ഭാവ "unary-ops.h"

സ്ഥിതം വരിയിൽ ദശാംശം op_abs(ദശാംശം x) {
    തിരികെ fabsf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_sgn(ദശാംശം x) {
    തിരികെ (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_neg(ദശാംശം x) {
    തിരികെ -x;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_step(ദശാംശം x) {
    തിരികെ (x > 0.f) ? 1.f : 0.f;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_tanh(ദശാംശം x) {
    തിരികെ tanhf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_elu(ദശാംശം x) {
    തിരികെ (x > 0.f) ? x : expm1f(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_relu(ദശാംശം x) {
    തിരികെ (x > 0.f) ? x : 0.f;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_sigmoid(ദശാംശം x) {
    തിരികെ 1.f / (1.f + expf(-x));
}

സ്ഥിതം വരിയിൽ ദശാംശം op_hardsigmoid(ദശാംശം x) {
    തിരികെ fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

സ്ഥിതം വരിയിൽ ദശാംശം op_exp(ദശാംശം x) {
    തിരികെ expf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_hardswish(ദശാംശം x) {
    തിരികെ x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

സ്ഥിതം വരിയിൽ ദശാംശം op_sqr(ദശാംശം x) {
    തിരികെ x * x;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_sqrt(ദശാംശം x) {
    തിരികെ sqrtf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_xielu(ദശാംശം x, ദശാംശം alpha_n, ദശാംശം alpha_p, ദശാംശം beta, ദശാംശം eps) {
    എങ്കിൽ (x > 0.0f) {
        തിരികെ alpha_p * x * x + beta * x;
    } അല്ലെങ്കിൽ {
        സ്ഥിരം ദശാംശം min_x_eps = fminf(x, eps);
        തിരികെ (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

സ്ഥിതം വരിയിൽ ദശാംശം op_sin(ദശാംശം x) {
    തിരികെ sinf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_cos(ദശാംശം x) {
    തിരികെ cosf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_log(ദശാംശം x) {
    തിരികെ logf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_expm1(ദശാംശം x) {
    തിരികെ expf(x) - 1.0f;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_softplus(ദശാംശം x) {
    തിരികെ (x > 20.0f) ? x : logf(1.0f + expf(x));
}

സ്ഥിതം വരിയിൽ ദശാംശം op_floor(ദശാംശം x) {
    തിരികെ floorf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_ceil(ദശാംശം x) {
    തിരികെ ceilf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_round(ദശാംശം x) {
    തിരികെ roundf(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം op_trunc(ദശാംശം x) {
    തിരികെ truncf(x);
}

template <ദശാംശം (*op)(ദശാംശം), typename src0_t, typename dst_t>
സ്ഥിതം വരിയിൽ ശൂന്യം vec_unary_op(int64_t n, dst_t * y, സ്ഥിരം src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <ദശാംശം (*op)(ദശാംശം), typename src0_t, typename dst_t>
സ്ഥിതം ശൂന്യം apply_unary_op(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == വലുപ്പം(dst_t));
    GGML_ASSERT(nb00 == വലുപ്പം(src0_t));

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, src0);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne02*ne01);
        സ്ഥിരം int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        സ്ഥിരം int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((അക്ഷരം *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        സ്ഥിരം src0_t * src0_ptr = (സ്ഥിരം src0_t *) ((സ്ഥിരം അക്ഷരം *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ദശാംശം (*op)(ദശാംശം)>
സ്ഥിതം ശൂന്യം unary_op(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    /*  */ എങ്കിൽ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ദശാംശം, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <ദശാംശം (*op)(ദശാംശം, ggml_tensor *)>
സ്ഥിതം ശൂന്യം unary_op_params(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    /*  */ എങ്കിൽ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, ദശാംശം, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
സ്ഥിതം വരിയിൽ ശൂന്യം vec_unary_op_functor(int64_t n, dst_t * y, സ്ഥിരം src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
സ്ഥിതം ശൂന്യം apply_unary_op_functor(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst, Op op) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == വലുപ്പം(dst_t));
    GGML_ASSERT(nb00 == വലുപ്പം(src0_t));

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, src0);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne02*ne01);
        സ്ഥിരം int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        സ്ഥിരം int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((അക്ഷരം *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        സ്ഥിരം src0_t * src0_ptr = (സ്ഥിരം src0_t *) ((സ്ഥിരം അക്ഷരം *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
സ്ഥിതം ശൂന്യം unary_op_functor(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst, Op op) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    /*  */ എങ്കിൽ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, ദശാംശം, ദശാംശം>(params, dst, op);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, ദശാംശം>(params, dst, op);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, ദശാംശം>(params, dst, op);
    } അല്ലെങ്കിൽ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

ശൂന്യം ggml_compute_forward_abs(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

ശൂന്യം ggml_compute_forward_sgn(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

ശൂന്യം ggml_compute_forward_neg(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

ശൂന്യം ggml_compute_forward_step(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

ശൂന്യം ggml_compute_forward_tanh(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

ശൂന്യം ggml_compute_forward_elu(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

ശൂന്യം ggml_compute_forward_relu(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

ശൂന്യം ggml_compute_forward_sigmoid(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

ശൂന്യം ggml_compute_forward_hardsigmoid(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

ശൂന്യം ggml_compute_forward_exp(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

ശൂന്യം ggml_compute_forward_hardswish(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

ശൂന്യം ggml_compute_forward_sqr(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

ശൂന്യം ggml_compute_forward_sqrt(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

ശൂന്യം ggml_compute_forward_sin(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

ശൂന്യം ggml_compute_forward_cos(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

ശൂന്യം ggml_compute_forward_log(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

ശൂന്യം ggml_compute_forward_expm1(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

ശൂന്യം ggml_compute_forward_softplus(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

ശൂന്യം ggml_compute_forward_floor(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

ശൂന്യം ggml_compute_forward_ceil(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

ശൂന്യം ggml_compute_forward_round(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

ശൂന്യം ggml_compute_forward_trunc(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

ശൂന്യം ggml_compute_forward_xielu(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ദശാംശം alpha_n = ggml_get_op_params_f32(dst, 1);
    സ്ഥിരം ദശാംശം alpha_p = ggml_get_op_params_f32(dst, 2);
    സ്ഥിരം ദശാംശം beta = ggml_get_op_params_f32(dst, 3);
    സ്ഥിരം ദശാംശം eps = ggml_get_op_params_f32(dst, 4);

    സ്ഥിരം auto xielu_op_params = [alpha_n, alpha_p, beta, eps](ദശാംശം f) {
        തിരികെ op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

