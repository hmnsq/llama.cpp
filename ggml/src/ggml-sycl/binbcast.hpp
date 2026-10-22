#अगर_अघोषित GGML_SYCL_BINBCAST_HPP
#घोषणा GGML_SYCL_BINBCAST_HPP
#समावेश "common.hpp"


अटल __dpct_inline__ भग्न op_repeat(स्थिर भग्न a, स्थिर भग्न b) {
    वापस b;
    GGML_UNUSED(a);
}

अटल __dpct_inline__ भग्न op_add(स्थिर भग्न a, स्थिर भग्न b) {
    वापस a + b;
}

अटल __dpct_inline__ भग्न op_sub(स्थिर भग्न a, स्थिर भग्न b) {
    वापस a - b;
}

अटल __dpct_inline__ भग्न op_mul(स्थिर भग्न a, स्थिर भग्न b) {
    वापस a * b;
}

अटल __dpct_inline__ भग्न op_div(स्थिर भग्न a, स्थिर भग्न b) {
    वापस a / b;
}

व्योम ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

व्योम ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
अंतरभूत bool ggml_sycl_add_kernel_supports(क्रमागत ggml_type src0, क्रमागत ggml_type src1, क्रमागत ggml_type dst) {
    अगर (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        वापस true;
    }
    अगर (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        वापस true;
    }
    अगर (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        वापस true;
    }
    अगर (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        वापस true;
    }
    अगर (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        वापस true;
    }
#अगर_घोषित GGML_SYCL_HAS_BF16
    अगर (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        वापस true;
    }
    अगर (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        वापस true;
    }
#पूर्ण_अगर
    वापस false;
}

#पूर्ण_अगर //GGML_SYCL_BINBCAST_HPP

