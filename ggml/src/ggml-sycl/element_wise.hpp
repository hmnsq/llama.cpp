#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_ELEMENTWISE_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_ELEMENTWISE_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ <limits> // For std::numeric_limits

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_GLU_ಖಂಡ_ಗಾತ್ರ 256

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
T neg_infinity() {
    ಹಿಂತಿರುಗಿ -std::numeric_limits<T>::infinity();
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T_Dst, ಬಗೆ_ಹೆಸರು T_Src = T_Dst>
ರಚನೆ typed_ದತ್ತಾಂಶ {
    ಸ್ಥಿರಾಂಕ T_Src * src;
    T_Dst * dst;
};

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T_Dst, ಬಗೆ_ಹೆಸರು T_Src = T_Dst>
typed_ದತ್ತಾಂಶ<T_Dst, T_Src> cast_ದತ್ತಾಂಶ(ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ {
        /* .src = */ static_cast<ಸ್ಥಿರಾಂಕ T_Src *>(dst->src[0]->ದತ್ತಾಂಶ),
        /* .dst = */ static_cast<T_Dst *>(dst->ದತ್ತಾಂಶ)
    };
}

ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ GELU_ಶೀಘ್ರ_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸ್ಥಿರ __dpct_inline__ T op_tanh(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#ಆದರೆ defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::tanh(x);
#ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ static_cast<T>(sycl::tanh(static_cast<ದಶಮಾಂಶ>(x)));
#ಆದರೆ_ಅಂತ್ಯ
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::tanh(x);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸ್ಥಿರ __dpct_inline__ T op_gelu(T x) {
    ಸ್ಥಿರಾಂಕ T GELU_COEF_A    = static_cast<T>(0.044715f);
    ಸ್ಥಿರಾಂಕ T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    ಹಿಂತಿರುಗಿ static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸ್ಥಿರ __dpct_inline__ T op_exp(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::ಘಾತಾಂಕ(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::ಘಾತಾಂಕ(x);
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸ್ಥಿರ __dpct_inline__ T op_silu(T x) {
    ಹಿಂತಿರುಗಿ x / (static_cast<T>(1.0f) + op_exp(-x));
}

ಶೂನ್ಯ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_gelu_ಶೀಘ್ರ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_ಹೆಜ್ಜೆ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_geglu_ಶೀಘ್ರ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_ಸುತ್ತು(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
ಶೂನ್ಯ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ಗಂಟು, ggml_tensor * mul_ಗಂಟು);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
ಶೂನ್ಯ ggml_sycl_op_ಕೂಡಿಸು_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * ಕೂಡಿಸು_ಗಂಟು,
                                      ggml_tensor * unary_ಗಂಟು, ggml_tensor * mul_ಗಂಟು);

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_ELEMENTWISE_HPP
