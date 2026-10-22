#اگر_غیر_تعریف_شدہ GGML_SYCL_ELEMENTWISE_HPP
#تعریف_کرو GGML_SYCL_ELEMENTWISE_HPP

#شامل "common.hpp"
#شامل "ggml.h"
#شامل <limits> // For std::numeric_limits

#تعریف_کرو SYCL_GLU_بلاک_سائز 256

سانچہ <قسم_نام T>
T neg_infinity() {
    واپس -std::numeric_limits<T>::infinity();
}

سانچہ<قسم_نام T_Dst, قسم_نام T_Src = T_Dst>
ساخت typed_ڈیٹا {
    ثابت T_Src * src;
    T_Dst * dst;
};

سانچہ<قسم_نام T_Dst, قسم_نام T_Src = T_Dst>
typed_ڈیٹا<T_Dst, T_Src> cast_ڈیٹا(ggml_tensor * dst) {
    واپس {
        /* .src = */ static_cast<ثابت T_Src *>(dst->src[0]->ڈیٹا),
        /* .dst = */ static_cast<T_Dst *>(dst->ڈیٹا)
    };
}

ثابت اعشاری GELU_تیز_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
سانچہ <قسم_نام T> ساکن __dpct_inline__ T op_tanh(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#اگر defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        واپس sycl::ext::oneapi::experimental::tanh(x);
#ورنہ
        واپس static_cast<T>(sycl::tanh(static_cast<اعشاری>(x)));
#اگر_ختم
    } ورنہ {
        واپس sycl::tanh(x);
    }
}

سانچہ <قسم_نام T> ساکن __dpct_inline__ T op_gelu(T x) {
    ثابت T GELU_COEF_A    = static_cast<T>(0.044715f);
    ثابت T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    واپس static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

سانچہ <قسم_نام T> ساکن __dpct_inline__ T op_exp(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        واپس sycl::ext::oneapi::experimental::قوت_نما(x);
    } ورنہ {
        واپس sycl::قوت_نما(x);
    }
}

سانچہ <قسم_نام T> ساکن __dpct_inline__ T op_silu(T x) {
    واپس x / (static_cast<T>(1.0f) + op_exp(-x));
}

خالی ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_gelu_تیز(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_قدم(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_geglu_تیز(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_باری(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
خالی ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
خالی ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_گرہ, ggml_tensor * mul_گرہ);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
خالی ggml_sycl_op_جوڑو_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * جوڑو_گرہ,
                                      ggml_tensor * unary_گرہ, ggml_tensor * mul_گرہ);

#اگر_ختم // GGML_SYCL_ELEMENTWISE_HPP
