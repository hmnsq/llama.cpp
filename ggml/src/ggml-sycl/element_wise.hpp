#अगर_अघोषित GGML_SYCL_ELEMENTWISE_HPP
#घोषणा GGML_SYCL_ELEMENTWISE_HPP

#समावेश "common.hpp"
#समावेश "ggml.h"
#समावेश <limits> // For std::numeric_limits

#घोषणा SYCL_GLU_BLOCK_आकार 256

ढाँचा <typename T>
T neg_infinity() {
    वापस -std::numeric_limits<T>::infinity();
}

ढाँचा<typename T_Dst, typename T_Src = T_Dst>
काष्ठा typed_तथ्य {
    स्थिर T_Src * src;
    T_Dst * dst;
};

ढाँचा<typename T_Dst, typename T_Src = T_Dst>
typed_तथ्य<T_Dst, T_Src> cast_तथ्य(ggml_tensor * dst) {
    वापस {
        /* .src = */ static_cast<स्थिर T_Src *>(dst->src[0]->तथ्य),
        /* .dst = */ static_cast<T_Dst *>(dst->तथ्य)
    };
}

स्थिर भग्न GELU_QUICK_COEF = -1.702f;

// Single-element activations, shared with the mat-vec kernels that fuse a GLU epilogue
// (mmvq.cpp), so both apply the same formula.
ढाँचा <typename T> अटल __dpct_inline__ T op_tanh(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
#अगर defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260000)
        वापस sycl::ext::oneapi::experimental::tanh(x);
#अन्यथा
        वापस static_cast<T>(sycl::tanh(static_cast<भग्न>(x)));
#पूर्ण_अगर
    } अन्यथा {
        वापस sycl::tanh(x);
    }
}

ढाँचा <typename T> अटल __dpct_inline__ T op_gelu(T x) {
    स्थिर T GELU_COEF_A    = static_cast<T>(0.044715f);
    स्थिर T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    वापस static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            op_tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

ढाँचा <typename T> अटल __dpct_inline__ T op_exp(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::exp(x);
    } अन्यथा {
        वापस sycl::exp(x);
    }
}

ढाँचा <typename T> अटल __dpct_inline__ T op_silu(T x) {
    वापस x / (static_cast<T>(1.0f) + op_exp(-x));
}

व्योम ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
व्योम ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
व्योम ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_गाँठ, ggml_tensor * mul_गाँठ);

// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
व्योम ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_गाँठ,
                                      ggml_tensor * unary_गाँठ, ggml_tensor * mul_गाँठ);

#पूर्ण_अगर // GGML_SYCL_ELEMENTWISE_HPP
