#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_ELEMENTWISE_HPP
#وصف_ڪريو GGML_SYCL_ELEMENTWISE_HPP

#ضمن "common.hpp"
#ضمن "ggml.h"
#ضمن <limits> // For std::numeric_limits

#وصف_ڪريو SYCL_GLU_بلاڪ_سائيز 256

سانچو <قسم_نالو T>
T neg_infinity() {
    أرجع -std::numeric_limits<T>::infinity();
}

سانچو<قسم_نالو T_Dst, قسم_نالو T_Src = T_Dst>
بنية typed_معلومات {
    ثابت T_Src * src;
    T_Dst * dst;
};

سانچو<قسم_نالو T_Dst, قسم_نالو T_Src = T_Dst>
typed_معلومات<T_Dst, T_Src> cast_معلومات(ggml_tensor * dst) {
    أرجع {
        /* .src = */ static_cast<ثابت T_Src *>(dst->src[0]->معلومات),
        /* .dst = */ static_cast<T_Dst *>(dst->معلومات)
    };
}

ثابت عائم GELU_تيز_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
سانچو <قسم_نالو T> ساكن __dpct_inline__ T op_tanh(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#إذا defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        أرجع sycl::ext::oneapi::experimental::tanh(x);
#وإلا
        أرجع static_cast<T>(sycl::tanh(static_cast<عائم>(x)));
#جيڪڏهن_پورو
    } وإلا {
        أرجع sycl::tanh(x);
    }
}

سانچو <قسم_نالو T> ساكن __dpct_inline__ T op_gelu(T x) {
    ثابت T GELU_COEF_A    = static_cast<T>(0.044715f);
    ثابت T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    أرجع static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

سانچو <قسم_نالو T> ساكن __dpct_inline__ T op_exp(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::قوت_نما(x);
    } وإلا {
        أرجع sycl::قوت_نما(x);
    }
}

سانچو <قسم_نالو T> ساكن __dpct_inline__ T op_silu(T x) {
    أرجع x / (static_cast<T>(1.0f) + op_exp(-x));
}

فراغ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_gelu_تيز(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

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

فراغ ggml_sycl_قدم(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

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
فراغ ggml_sycl_geglu_تيز(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_ڦيرو(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
فراغ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
فراغ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ڳنڍ, ggml_tensor * mul_ڳنڍ);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
فراغ ggml_sycl_op_جوڙيو_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * جوڙيو_ڳنڍ,
                                      ggml_tensor * unary_ڳنڍ, ggml_tensor * mul_ڳنڍ);

#جيڪڏهن_پورو // GGML_SYCL_ELEMENTWISE_HPP
