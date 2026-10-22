#यदि_अपरिभाषितम् GGML_SYCL_BINBCAST_HPP
#परिभाषय GGML_SYCL_BINBCAST_HPP
#अन्तर्भाव "common.hpp"


स्थिर __dpct_inline__ भिन्न op_repeat(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    प्रत्यागम b;
    GGML_अप्रयुक्तम्(a);
}

स्थिर __dpct_inline__ भिन्न op_सङ्कलय(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    प्रत्यागम a + b;
}

स्थिर __dpct_inline__ भिन्न op_sub(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    प्रत्यागम a - b;
}

स्थिर __dpct_inline__ भिन्न op_mul(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    प्रत्यागम a * b;
}

स्थिर __dpct_inline__ भिन्न op_div(ध्रुव भिन्न a, ध्रुव भिन्न b) {
    प्रत्यागम a / b;
}

शून्य ggml_sycl_सङ्कलय(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_op_सङ्कलय_सङ्कलय_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
अन्तःस्थ तार्किक ggml_sycl_सङ्कलय_kernel_supports(गणना ggml_type src0, गणना ggml_type src1, गणना ggml_type dst) {
    यदि (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        प्रत्यागम सत्य;
    }
    यदि (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        प्रत्यागम सत्य;
    }
    यदि (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        प्रत्यागम सत्य;
    }
    यदि (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        प्रत्यागम सत्य;
    }
    यदि (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        प्रत्यागम सत्य;
    }
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
    यदि (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        प्रत्यागम सत्य;
    }
    यदि (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        प्रत्यागम सत्य;
    }
#यदि_अन्तः
    प्रत्यागम असत्य;
}

#यदि_अन्तः //GGML_SYCL_BINBCAST_HPP

