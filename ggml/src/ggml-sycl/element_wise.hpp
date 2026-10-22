#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_ELEMENTWISE_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_ELEMENTWISE_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <limits> // For std::numeric_limits

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_GLU_ਖੰਡ_ਮਾਪ 256

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
T neg_infinity() {
    ਵਾਪਸ -std::numeric_limits<T>::infinity();
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T_Dst, ਕਿਸਮ_ਨਾਂ T_Src = T_Dst>
ਬਣਤਰ typed_ਅੰਕੜੇ {
    ਅਚਲ T_Src * src;
    T_Dst * dst;
};

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T_Dst, ਕਿਸਮ_ਨਾਂ T_Src = T_Dst>
typed_ਅੰਕੜੇ<T_Dst, T_Src> cast_ਅੰਕੜੇ(ggml_tensor * dst) {
    ਵਾਪਸ {
        /* .src = */ static_cast<ਅਚਲ T_Src *>(dst->src[0]->ਅੰਕੜੇ),
        /* .dst = */ static_cast<T_Dst *>(dst->ਅੰਕੜੇ)
    };
}

ਅਚਲ ਦਸ਼ਮਲਵ GELU_ਤੇਜ਼_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਥਿਰ __dpct_inline__ T op_tanh(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#ਜੇ defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        ਵਾਪਸ sycl::ext::oneapi::experimental::tanh(x);
#ਨਹੀਂਤਾਂ
        ਵਾਪਸ static_cast<T>(sycl::tanh(static_cast<ਦਸ਼ਮਲਵ>(x)));
#ਜੇ_ਅੰਤ
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::tanh(x);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਥਿਰ __dpct_inline__ T op_gelu(T x) {
    ਅਚਲ T GELU_COEF_A    = static_cast<T>(0.044715f);
    ਅਚਲ T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    ਵਾਪਸ static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਥਿਰ __dpct_inline__ T op_exp(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::ਘਾਤਾਂਕ(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::ਘਾਤਾਂਕ(x);
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਥਿਰ __dpct_inline__ T op_silu(T x) {
    ਵਾਪਸ x / (static_cast<T>(1.0f) + op_exp(-x));
}

ਖਾਲੀ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_gelu_ਤੇਜ਼(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_ਕਦਮ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_geglu_ਤੇਜ਼(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_ਗੇੜਾ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ਖਾਲੀ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
ਖਾਲੀ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ਗੰਢ, ggml_tensor * mul_ਗੰਢ);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
ਖਾਲੀ ggml_sycl_op_ਜੋੜੋ_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * ਜੋੜੋ_ਗੰਢ,
                                      ggml_tensor * unary_ਗੰਢ, ggml_tensor * mul_ਗੰਢ);

#ਜੇ_ਅੰਤ // GGML_SYCL_ELEMENTWISE_HPP
