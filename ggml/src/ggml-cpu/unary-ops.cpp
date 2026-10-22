#include "unary-ops.h"

static anta_hstha float op_abs(float x) {
    ສົ່ງຄືນ fabsf(x);
}

static anta_hstha float op_sgn(float x) {
    ສົ່ງຄືນ (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static anta_hstha float op_neg(float x) {
    ສົ່ງຄືນ -x;
}

static anta_hstha float op_step(float x) {
    ສົ່ງຄືນ (x > 0.f) ? 1.f : 0.f;
}

static anta_hstha float op_tanh(float x) {
    ສົ່ງຄືນ tanhf(x);
}

static anta_hstha float op_elu(float x) {
    ສົ່ງຄືນ (x > 0.f) ? x : expm1f(x);
}

static anta_hstha float op_relu(float x) {
    ສົ່ງຄືນ (x > 0.f) ? x : 0.f;
}

static anta_hstha float op_sigmoid(float x) {
    ສົ່ງຄືນ 1.f / (1.f + expf(-x));
}

static anta_hstha float op_hardsigmoid(float x) {
    ສົ່ງຄືນ fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static anta_hstha float op_exp(float x) {
    ສົ່ງຄືນ expf(x);
}

static anta_hstha float op_hardswish(float x) {
    ສົ່ງຄືນ x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static anta_hstha float op_sqr(float x) {
    ສົ່ງຄືນ x * x;
}

static anta_hstha float op_sqrt(float x) {
    ສົ່ງຄືນ sqrtf(x);
}

static anta_hstha float op_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    ຖ້າ (x > 0.0f) {
        ສົ່ງຄືນ alpha_p * x * x + beta * x;
    } ບໍ່ດັ່ງນັ້ນ {
        const float min_x_eps = fminf(x, eps);
        ສົ່ງຄືນ (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

static anta_hstha float op_sin(float x) {
    ສົ່ງຄືນ sinf(x);
}

static anta_hstha float op_cos(float x) {
    ສົ່ງຄືນ cosf(x);
}

static anta_hstha float op_log(float x) {
    ສົ່ງຄືນ logf(x);
}

static anta_hstha float op_expm1(float x) {
    ສົ່ງຄືນ expf(x) - 1.0f;
}

static anta_hstha float op_softplus(float x) {
    ສົ່ງຄືນ (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static anta_hstha float op_floor(float x) {
    ສົ່ງຄືນ floorf(x);
}

static anta_hstha float op_ceil(float x) {
    ສົ່ງຄືນ ceilf(x);
}

static anta_hstha float op_round(float x) {
    ສົ່ງຄືນ roundf(x);
}

static anta_hstha float op_trunc(float x) {
    ສົ່ງຄືນ truncf(x);
}

template <float (*op)(float), typename src0_t, typename dst_t>
static anta_hstha void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ສຳລັບ (ຈຳນວນເຕັມ i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == parimaa_na(dst_t));
    GGML_ASSERT(nb00 == parimaa_na(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    ສຳລັບ (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((var_na *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const var_na *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ ຖ້າ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <float (*op)(float, ggml_tensor *)>
static void unary_op_params(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ ຖ້າ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } ບໍ່ດັ່ງນັ້ນ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static anta_hstha void vec_unary_op_functor(int64_t n, dst_t * y, const src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ສຳລັບ (ຈຳນວນເຕັມ i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static void apply_unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == parimaa_na(dst_t));
    GGML_ASSERT(nb00 == parimaa_na(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    ສຳລັບ (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((var_na *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const var_na *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
static void unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ ຖ້າ (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, float, float>(params, dst, op);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, float>(params, dst, op);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, float>(params, dst, op);
    } ບໍ່ດັ່ງນັ້ນ {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_compute_forward_abs(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

void ggml_compute_forward_neg(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

void ggml_compute_forward_step(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

void ggml_compute_forward_tanh(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

void ggml_compute_forward_elu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

void ggml_compute_forward_relu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

void ggml_compute_forward_sigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

void ggml_compute_forward_hardsigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

void ggml_compute_forward_exp(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

void ggml_compute_forward_hardswish(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

void ggml_compute_forward_sqr(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_sin(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

void ggml_compute_forward_log(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

void ggml_compute_forward_expm1(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

void ggml_compute_forward_softplus(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

void ggml_compute_forward_floor(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

void ggml_compute_forward_ceil(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

void ggml_compute_forward_round(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

void ggml_compute_forward_trunc(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

void ggml_compute_forward_xielu(const ggml_compute_params * params, ggml_tensor * dst) {
    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta = ggml_get_op_params_f32(dst, 3);
    const float eps = ggml_get_op_params_f32(dst, 4);

    const auto xielu_op_params = [alpha_n, alpha_p, beta, eps](float f) {
        ສົ່ງຄືນ op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

