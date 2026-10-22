#అన్తర్భావ "unary-ops.h"

స్థిర అంతఃవరుస దశాంశం op_abs(దశాంశం x) {
    తిరిగి fabsf(x);
}

స్థిర అంతఃవరుస దశాంశం op_sgn(దశాంశం x) {
    తిరిగి (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

స్థిర అంతఃవరుస దశాంశం op_neg(దశాంశం x) {
    తిరిగి -x;
}

స్థిర అంతఃవరుస దశాంశం op_అడుగు(దశాంశం x) {
    తిరిగి (x > 0.f) ? 1.f : 0.f;
}

స్థిర అంతఃవరుస దశాంశం op_tanh(దశాంశం x) {
    తిరిగి tanhf(x);
}

స్థిర అంతఃవరుస దశాంశం op_elu(దశాంశం x) {
    తిరిగి (x > 0.f) ? x : expm1f(x);
}

స్థిర అంతఃవరుస దశాంశం op_relu(దశాంశం x) {
    తిరిగి (x > 0.f) ? x : 0.f;
}

స్థిర అంతఃవరుస దశాంశం op_sigmoid(దశాంశం x) {
    తిరిగి 1.f / (1.f + expf(-x));
}

స్థిర అంతఃవరుస దశాంశం op_hardsigmoid(దశాంశం x) {
    తిరిగి fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

స్థిర అంతఃవరుస దశాంశం op_exp(దశాంశం x) {
    తిరిగి expf(x);
}

స్థిర అంతఃవరుస దశాంశం op_hardswish(దశాంశం x) {
    తిరిగి x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

స్థిర అంతఃవరుస దశాంశం op_sqr(దశాంశం x) {
    తిరిగి x * x;
}

స్థిర అంతఃవరుస దశాంశం op_sqrt(దశాంశం x) {
    తిరిగి sqrtf(x);
}

స్థిర అంతఃవరుస దశాంశం op_xielu(దశాంశం x, దశాంశం alpha_n, దశాంశం alpha_p, దశాంశం beta, దశాంశం eps) {
    ఒకవేళ (x > 0.0f) {
        తిరిగి alpha_p * x * x + beta * x;
    } లేకపోతే {
        స్థిరరాశి దశాంశం min_x_eps = fminf(x, eps);
        తిరిగి (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

స్థిర అంతఃవరుస దశాంశం op_sin(దశాంశం x) {
    తిరిగి sinf(x);
}

స్థిర అంతఃవరుస దశాంశం op_cos(దశాంశం x) {
    తిరిగి cosf(x);
}

స్థిర అంతఃవరుస దశాంశం op_log(దశాంశం x) {
    తిరిగి logf(x);
}

స్థిర అంతఃవరుస దశాంశం op_expm1(దశాంశం x) {
    తిరిగి expf(x) - 1.0f;
}

స్థిర అంతఃవరుస దశాంశం op_softplus(దశాంశం x) {
    తిరిగి (x > 20.0f) ? x : logf(1.0f + expf(x));
}

స్థిర అంతఃవరుస దశాంశం op_floor(దశాంశం x) {
    తిరిగి floorf(x);
}

స్థిర అంతఃవరుస దశాంశం op_ceil(దశాంశం x) {
    తిరిగి ceilf(x);
}

స్థిర అంతఃవరుస దశాంశం op_విడత(దశాంశం x) {
    తిరిగి roundf(x);
}

స్థిర అంతఃవరుస దశాంశం op_trunc(దశాంశం x) {
    తిరిగి truncf(x);
}

మూస <దశాంశం (*op)(దశాంశం), రకం_పేరు src0_t, రకం_పేరు dst_t>
స్థిర అంతఃవరుస శూన్య vec_unary_op(int64_t n, dst_t * y, స్థిరరాశి src0_t * x) {
    constexpr స్వయంచాలక src0_గమ్యం_f32 = type_conversion_పట్టిక<src0_t>::గమ్యం_f32;
    constexpr స్వయంచాలక f32_గమ్యం_dst  = type_conversion_పట్టిక<dst_t >::from_f32;

    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        y[i] = f32_గమ్యం_dst(op(src0_గమ్యం_f32(x[i])));
    }
}

మూస <దశాంశం (*op)(దశాంశం), రకం_పేరు src0_t, రకం_పేరు dst_t>
స్థిర శూన్య apply_unary_op(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src0) && ggml_is_contiguous_అడ్డువరుసలు(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == పరిమాణం(dst_t));
    GGML_ASSERT(nb00 == పరిమాణం(src0_t));

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, src0);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne02*ne01);
        స్థిరరాశి int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        స్థిరరాశి int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((అక్షరం *)       dst->దత్తాంశం  + i03*nb3  + i02*nb2  + i01*nb1 );
        స్థిరరాశి src0_t * src0_ptr = (స్థిరరాశి src0_t *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
మూస <దశాంశం (*op)(దశాంశం)>
స్థిర శూన్య unary_op(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    /*  */ ఒకవేళ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, దశాంశం, దశాంశం>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, దశాంశం>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, దశాంశం>(params, dst);
    } లేకపోతే {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

మూస <దశాంశం (*op)(దశాంశం, ggml_tensor *)>
స్థిర శూన్య unary_op_params(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    /*  */ ఒకవేళ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, దశాంశం, దశాంశం>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, దశాంశం>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, దశాంశం>(params, dst);
    } లేకపోతే {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
మూస <రకం_పేరు Op, రకం_పేరు src0_t, రకం_పేరు dst_t>
స్థిర అంతఃవరుస శూన్య vec_unary_op_functor(int64_t n, dst_t * y, స్థిరరాశి src0_t * x, Op op) {
    constexpr స్వయంచాలక src0_గమ్యం_f32 = type_conversion_పట్టిక<src0_t>::గమ్యం_f32;
    constexpr స్వయంచాలక f32_గమ్యం_dst  = type_conversion_పట్టిక<dst_t >::from_f32;

    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        y[i] = f32_గమ్యం_dst(op(src0_గమ్యం_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
మూస <రకం_పేరు Op, రకం_పేరు src0_t, రకం_పేరు dst_t>
స్థిర శూన్య apply_unary_op_functor(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst, Op op) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == పరిమాణం(dst_t));
    GGML_ASSERT(nb00 == పరిమాణం(src0_t));

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, src0);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne02*ne01);
        స్థిరరాశి int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        స్థిరరాశి int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((అక్షరం *)       dst->దత్తాంశం  + i03*nb3  + i02*nb2  + i01*nb1 );
        స్థిరరాశి src0_t * src0_ptr = (స్థిరరాశి src0_t *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
మూస <రకం_పేరు Op>
స్థిర శూన్య unary_op_functor(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst, Op op) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    /*  */ ఒకవేళ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, దశాంశం, దశాంశం>(params, dst, op);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, దశాంశం>(params, dst, op);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, దశాంశం>(params, dst, op);
    } లేకపోతే {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

శూన్య ggml_compute_forward_abs(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

శూన్య ggml_compute_forward_sgn(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

శూన్య ggml_compute_forward_neg(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

శూన్య ggml_compute_forward_అడుగు(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_అడుగు>(params, dst);
}

శూన్య ggml_compute_forward_tanh(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

శూన్య ggml_compute_forward_elu(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

శూన్య ggml_compute_forward_relu(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

శూన్య ggml_compute_forward_sigmoid(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

శూన్య ggml_compute_forward_hardsigmoid(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

శూన్య ggml_compute_forward_exp(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

శూన్య ggml_compute_forward_hardswish(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

శూన్య ggml_compute_forward_sqr(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

శూన్య ggml_compute_forward_sqrt(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

శూన్య ggml_compute_forward_sin(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

శూన్య ggml_compute_forward_cos(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

శూన్య ggml_compute_forward_log(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

శూన్య ggml_compute_forward_expm1(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

శూన్య ggml_compute_forward_softplus(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

శూన్య ggml_compute_forward_floor(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

శూన్య ggml_compute_forward_ceil(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

శూన్య ggml_compute_forward_విడత(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_విడత>(params, dst);
}

శూన్య ggml_compute_forward_trunc(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

శూన్య ggml_compute_forward_xielu(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి దశాంశం alpha_n = ggml_get_op_params_f32(dst, 1);
    స్థిరరాశి దశాంశం alpha_p = ggml_get_op_params_f32(dst, 2);
    స్థిరరాశి దశాంశం beta = ggml_get_op_params_f32(dst, 3);
    స్థిరరాశి దశాంశం eps = ggml_get_op_params_f32(dst, 4);

    స్థిరరాశి స్వయంచాలక xielu_op_params = [alpha_n, alpha_p, beta, eps](దశాంశం f) {
        తిరిగి op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

