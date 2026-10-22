#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#ಅನ್ತರ್ಭಾವ "common.hpp"


ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ op_repeat(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ b;
    GGML_UNUSED(a);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ op_add(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a + b;
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ op_sub(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a - b;
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ op_mul(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a * b;
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ op_div(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) {
    ಪ್ರತ್ಯಾಗಮ a / b;
}

ಶೂನ್ಯ ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_sycl_add_kernel_supports(ಗಣನಾ ggml_type src0, ಗಣನಾ ggml_type src1, ಗಣನಾ ggml_type dst) {
    ಯದಿ (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಯದಿ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಯದಿ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಯದಿ (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಯದಿ (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
#ifdef GGML_SYCL_HAS_BF16
    ಯದಿ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಯದಿ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
#endif
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

#endif //GGML_SYCL_BINBCAST_HPP

