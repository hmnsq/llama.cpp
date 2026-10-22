#నిర్వచితం_కాకపోతే GGML_SYCL_ELEMENTWISE_HPP
#నిర్వచించు GGML_SYCL_ELEMENTWISE_HPP

#అన్తర్భావ "common.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ <limits> // For std::numeric_limits

#నిర్వచించు SYCL_GLU_ఖండం_కొలత 256

మూస <రకం_పేరు T>
T neg_infinity() {
    తిరిగి -std::numeric_limits<T>::infinity();
}

మూస<రకం_పేరు T_Dst, రకం_పేరు T_Src = T_Dst>
నిర్మాణం typed_దత్తాంశం {
    స్థిరరాశి T_Src * src;
    T_Dst * dst;
};

మూస<రకం_పేరు T_Dst, రకం_పేరు T_Src = T_Dst>
typed_దత్తాంశం<T_Dst, T_Src> cast_దత్తాంశం(ggml_tensor * dst) {
    తిరిగి {
        /* .src = */ static_cast<స్థిరరాశి T_Src *>(dst->src[0]->దత్తాంశం),
        /* .dst = */ static_cast<T_Dst *>(dst->దత్తాంశం)
    };
}

స్థిరరాశి దశాంశం GELU_శీఘ్ర_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
మూస <రకం_పేరు T> స్థిర __dpct_inline__ T op_tanh(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#ఒకవేళ defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        తిరిగి sycl::ext::oneapi::experimental::tanh(x);
#లేకపోతే
        తిరిగి static_cast<T>(sycl::tanh(static_cast<దశాంశం>(x)));
#అయితే_ముగింపు
    } లేకపోతే {
        తిరిగి sycl::tanh(x);
    }
}

మూస <రకం_పేరు T> స్థిర __dpct_inline__ T op_gelu(T x) {
    స్థిరరాశి T GELU_COEF_A    = static_cast<T>(0.044715f);
    స్థిరరాశి T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    తిరిగి static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

మూస <రకం_పేరు T> స్థిర __dpct_inline__ T op_exp(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::ఘాతాంకం(x);
    } లేకపోతే {
        తిరిగి sycl::ఘాతాంకం(x);
    }
}

మూస <రకం_పేరు T> స్థిర __dpct_inline__ T op_silu(T x) {
    తిరిగి x / (static_cast<T>(1.0f) + op_exp(-x));
}

శూన్య ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_gelu_శీఘ్ర(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_అడుగు(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_geglu_శీఘ్ర(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_విడత(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
శూన్య ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
శూన్య ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_కణుపు, ggml_tensor * mul_కణుపు);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
శూన్య ggml_sycl_op_కూడు_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * కూడు_కణుపు,
                                      ggml_tensor * unary_కణుపు, ggml_tensor * mul_కణుపు);

#అయితే_ముగింపు // GGML_SYCL_ELEMENTWISE_HPP
