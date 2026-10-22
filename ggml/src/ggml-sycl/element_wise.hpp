#ifndef GGML_SYCL_ELEMENTWISE_HPP
#define GGML_SYCL_ELEMENTWISE_HPP

#অন্তর্ভাৱ "common.hpp"
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ <limits> // For std::numeric_limits

#define SYCL_GLU_BLOCK_SIZE 256

template <typename T>
T neg_infinity() {
    ঘূৰাই -std::numeric_limits<T>::infinity();
}

template<typename T_Dst, typename T_Src = T_Dst>
সংরচনা typed_data {
    ধ্রুৱ T_Src * src;
    T_Dst * dst;
};

template<typename T_Dst, typename T_Src = T_Dst>
typed_data<T_Dst, T_Src> cast_data(ggml_tensor * dst) {
    ঘূৰাই {
        /* .src = */ static_cast<ধ্রুৱ T_Src *>(dst->src[0]->data),
        /* .dst = */ static_cast<T_Dst *>(dst->data)
    };
}

ধ্রুৱ ভিন্ন GELU_QUICK_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
template <typename T> স্থির __dpct_inline__ T op_tanh(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#যদি defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        ঘূৰাই sycl::ext::oneapi::experimental::tanh(x);
#অন্যথা
        ঘূৰাই static_cast<T>(sycl::tanh(static_cast<ভিন্ন>(x)));
#endif
    } অন্যথা {
        ঘূৰাই sycl::tanh(x);
    }
}

template <typename T> স্থির __dpct_inline__ T op_gelu(T x) {
    ধ্রুৱ T GELU_COEF_A    = static_cast<T>(0.044715f);
    ধ্রুৱ T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    ঘূৰাই static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

template <typename T> স্থির __dpct_inline__ T op_exp(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ঘূৰাই sycl::ext::oneapi::experimental::exp(x);
    } অন্যথা {
        ঘূৰাই sycl::exp(x);
    }
}

template <typename T> স্থির __dpct_inline__ T op_silu(T x) {
    ঘূৰাই x / (static_cast<T>(1.0f) + op_exp(-x));
}

শূন্য ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
শূন্য ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
শূন্য ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_গ্ৰন্থি, ggml_tensor * mul_গ্ৰন্থি);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
শূন্য ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_গ্ৰন্থি,
                                      ggml_tensor * unary_গ্ৰন্থি, ggml_tensor * mul_গ্ৰন্থি);

#endif // GGML_SYCL_ELEMENTWISE_HPP
