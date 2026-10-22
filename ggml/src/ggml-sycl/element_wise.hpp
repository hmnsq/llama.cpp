#ifndef GGML_SYCL_ELEMENTWISE_HPP
#define GGML_SYCL_ELEMENTWISE_HPP

#ضمن "common.hpp"
#ضمن "ggml.h"
#ضمن <limits> // For std::numeric_limits

#define SYCL_GLU_BLOCK_SIZE 256

template <typename T>
T neg_infinity() {
    أرجع -std::numeric_limits<T>::infinity();
}

template<typename T_Dst, typename T_Src = T_Dst>
بنية typed_data {
    ثابت T_Src * src;
    T_Dst * dst;
};

template<typename T_Dst, typename T_Src = T_Dst>
typed_data<T_Dst, T_Src> cast_data(ggml_tensor * dst) {
    أرجع {
        /* .src = */ static_cast<ثابت T_Src *>(dst->src[0]->data),
        /* .dst = */ static_cast<T_Dst *>(dst->data)
    };
}

ثابت عائم GELU_QUICK_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
template <typename T> ساكن __dpct_inline__ T op_tanh(T x) {
    𐡄𐡍 constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#𐡄𐡍 defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        أرجع sycl::ext::oneapi::experimental::tanh(x);
#وإلا
        أرجع static_cast<T>(sycl::tanh(static_cast<عائم>(x)));
#endif
    } وإلا {
        أرجع sycl::tanh(x);
    }
}

template <typename T> ساكن __dpct_inline__ T op_gelu(T x) {
    ثابت T GELU_COEF_A    = static_cast<T>(0.044715f);
    ثابت T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    أرجع static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

template <typename T> ساكن __dpct_inline__ T op_exp(T x) {
    𐡄𐡍 constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::exp(x);
    } وإلا {
        أرجع sycl::exp(x);
    }
}

template <typename T> ساكن __dpct_inline__ T op_silu(T x) {
    أرجع x / (static_cast<T>(1.0f) + op_exp(-x));
}

فراغ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
فراغ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
فراغ ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_node,
                                      ggml_tensor * unary_node, ggml_tensor * mul_node);

#endif // GGML_SYCL_ELEMENTWISE_HPP
