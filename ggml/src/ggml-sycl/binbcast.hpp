#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_BINBCAST_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_BINBCAST_HPP
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"


ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ op_repeat(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ b;
    GGML_ਅਣਵਰਤਿਆ(a);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ op_ਜੋੜੋ(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a + b;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ op_sub(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a - b;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ op_mul(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a * b;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ op_div(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) {
    ਵਾਪਸ a / b;
}

ਖਾਲੀ ggml_sycl_ਜੋੜੋ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ਖਾਲੀ ggml_sycl_op_ਜੋੜੋ_ਜੋੜੋ_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_ਜੋੜੋ_kernel_supports(ਗਿਣਤੀ ggml_type src0, ਗਿਣਤੀ ggml_type src1, ਗਿਣਤੀ ggml_type dst) {
    ਜੇ (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        ਵਾਪਸ ਸੱਚ;
    }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    ਜੇ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        ਵਾਪਸ ਸੱਚ;
    }
#ਜੇ_ਅੰਤ
    ਵਾਪਸ ਝੂਠ;
}

#ਜੇ_ਅੰਤ //GGML_SYCL_BINBCAST_HPP

