#અન્તર્ભાવ "unary-ops.h"

સ્થિર અંતઃરેખ દશાંશ op_abs(દશાંશ x) {
    પાછા fabsf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_sgn(દશાંશ x) {
    પાછા (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

સ્થિર અંતઃરેખ દશાંશ op_neg(દશાંશ x) {
    પાછા -x;
}

સ્થિર અંતઃરેખ દશાંશ op_step(દશાંશ x) {
    પાછા (x > 0.f) ? 1.f : 0.f;
}

સ્થિર અંતઃરેખ દશાંશ op_tanh(દશાંશ x) {
    પાછા tanhf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_elu(દશાંશ x) {
    પાછા (x > 0.f) ? x : expm1f(x);
}

સ્થિર અંતઃરેખ દશાંશ op_relu(દશાંશ x) {
    પાછા (x > 0.f) ? x : 0.f;
}

સ્થિર અંતઃરેખ દશાંશ op_sigmoid(દશાંશ x) {
    પાછા 1.f / (1.f + expf(-x));
}

સ્થિર અંતઃરેખ દશાંશ op_hardsigmoid(દશાંશ x) {
    પાછા fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

સ્થિર અંતઃરેખ દશાંશ op_exp(દશાંશ x) {
    પાછા expf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_hardswish(દશાંશ x) {
    પાછા x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

સ્થિર અંતઃરેખ દશાંશ op_sqr(દશાંશ x) {
    પાછા x * x;
}

સ્થિર અંતઃરેખ દશાંશ op_sqrt(દશાંશ x) {
    પાછા sqrtf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_xielu(દશાંશ x, દશાંશ alpha_n, દશાંશ alpha_p, દશાંશ beta, દશાંશ eps) {
    જો (x > 0.0f) {
        પાછા alpha_p * x * x + beta * x;
    } નહીંતર {
        અચલ દશાંશ min_x_eps = fminf(x, eps);
        પાછા (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

સ્થિર અંતઃરેખ દશાંશ op_sin(દશાંશ x) {
    પાછા sinf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_cos(દશાંશ x) {
    પાછા cosf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_log(દશાંશ x) {
    પાછા logf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_expm1(દશાંશ x) {
    પાછા expf(x) - 1.0f;
}

સ્થિર અંતઃરેખ દશાંશ op_softplus(દશાંશ x) {
    પાછા (x > 20.0f) ? x : logf(1.0f + expf(x));
}

સ્થિર અંતઃરેખ દશાંશ op_floor(દશાંશ x) {
    પાછા floorf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_ceil(દશાંશ x) {
    પાછા ceilf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_round(દશાંશ x) {
    પાછા roundf(x);
}

સ્થિર અંતઃરેખ દશાંશ op_trunc(દશાંશ x) {
    પાછા truncf(x);
}

template <દશાંશ (*op)(દશાંશ), typename src0_t, typename dst_t>
સ્થિર અંતઃરેખ શૂન્ય vec_unary_op(int64_t n, dst_t * y, અચલ src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <દશાંશ (*op)(દશાંશ), typename src0_t, typename dst_t>
સ્થિર શૂન્ય apply_unary_op(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == કદ(dst_t));
    GGML_ASSERT(nb00 == કદ(src0_t));

    અચલ auto [ir0, ir1] = get_thread_range(params, src0);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne02*ne01);
        અચલ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        અચલ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((અક્ષર *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        અચલ src0_t * src0_ptr = (અચલ src0_t *) ((અચલ અક્ષર *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <દશાંશ (*op)(દશાંશ)>
સ્થિર શૂન્ય unary_op(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    /*  */ જો (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, દશાંશ, દશાંશ>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, દશાંશ>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, દશાંશ>(params, dst);
    } નહીંતર {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <દશાંશ (*op)(દશાંશ, ggml_tensor *)>
સ્થિર શૂન્ય unary_op_params(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    /*  */ જો (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, દશાંશ, દશાંશ>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, દશાંશ>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, દશાંશ>(params, dst);
    } નહીંતર {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
સ્થિર અંતઃરેખ શૂન્ય vec_unary_op_functor(int64_t n, dst_t * y, અચલ src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
સ્થિર શૂન્ય apply_unary_op_functor(અચલ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == કદ(dst_t));
    GGML_ASSERT(nb00 == કદ(src0_t));

    અચલ auto [ir0, ir1] = get_thread_range(params, src0);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne02*ne01);
        અચલ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        અચલ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((અક્ષર *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        અચલ src0_t * src0_ptr = (અચલ src0_t *) ((અચલ અક્ષર *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
સ્થિર શૂન્ય unary_op_functor(અચલ ggml_compute_params * params, ggml_tensor * dst, Op op) {
    અચલ ggml_tensor * src0 = dst->src[0];

    /*  */ જો (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, દશાંશ, દશાંશ>(params, dst, op);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, દશાંશ>(params, dst, op);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, દશાંશ>(params, dst, op);
    } નહીંતર {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

શૂન્ય ggml_compute_forward_abs(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

શૂન્ય ggml_compute_forward_sgn(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

શૂન્ય ggml_compute_forward_neg(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

શૂન્ય ggml_compute_forward_step(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

શૂન્ય ggml_compute_forward_tanh(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

શૂન્ય ggml_compute_forward_elu(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

શૂન્ય ggml_compute_forward_relu(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

શૂન્ય ggml_compute_forward_sigmoid(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

શૂન્ય ggml_compute_forward_hardsigmoid(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

શૂન્ય ggml_compute_forward_exp(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

શૂન્ય ggml_compute_forward_hardswish(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

શૂન્ય ggml_compute_forward_sqr(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

શૂન્ય ggml_compute_forward_sqrt(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

શૂન્ય ggml_compute_forward_sin(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

શૂન્ય ggml_compute_forward_cos(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

શૂન્ય ggml_compute_forward_log(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

શૂન્ય ggml_compute_forward_expm1(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

શૂન્ય ggml_compute_forward_softplus(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

શૂન્ય ggml_compute_forward_floor(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

શૂન્ય ggml_compute_forward_ceil(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

શૂન્ય ggml_compute_forward_round(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

શૂન્ય ggml_compute_forward_trunc(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

શૂન્ય ggml_compute_forward_xielu(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ દશાંશ alpha_n = ggml_get_op_params_f32(dst, 1);
    અચલ દશાંશ alpha_p = ggml_get_op_params_f32(dst, 2);
    અચલ દશાંશ beta = ggml_get_op_params_f32(dst, 3);
    અચલ દશાંશ eps = ggml_get_op_params_f32(dst, 4);

    અચલ auto xielu_op_params = [alpha_n, alpha_p, beta, eps](દશાંશ f) {
        પાછા op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

