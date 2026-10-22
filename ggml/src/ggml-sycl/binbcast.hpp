#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#अन्तर्भाव "common.hpp"


स्थिर __dpct_inline__ भिन्न op_repeat(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    पटिनिवत्त b;
    GGML_UNUSED(a);
}

स्थिर __dpct_inline__ भिन्न op_add(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    पटिनिवत्त a + b;
}

स्थिर __dpct_inline__ भिन्न op_sub(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    पटिनिवत्त a - b;
}

स्थिर __dpct_inline__ भिन्न op_mul(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    पटिनिवत्त a * b;
}

स्थिर __dpct_inline__ भिन्न op_div(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    पटिनिवत्त a / b;
}

शून्य ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
अन्तःस्थ तार्किक ggml_sycl_add_kernel_supports(गणना ggml_type src0, गणना ggml_type src1, गणना ggml_type dst) {
    सचे (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        पटिनिवत्त सच्च;
    }
    सचे (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        पटिनिवत्त सच्च;
    }
    सचे (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        पटिनिवत्त सच्च;
    }
    सचे (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        पटिनिवत्त सच्च;
    }
    सचे (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        पटिनिवत्त सच्च;
    }
#ifdef GGML_SYCL_HAS_BF16
    सचे (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        पटिनिवत्त सच्च;
    }
    सचे (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        पटिनिवत्त सच्च;
    }
#endif
    पटिनिवत्त मिच्छा;
}

#endif //GGML_SYCL_BINBCAST_HPP

