#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_BINBCAST_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_BINBCAST_HPP
#ಅನ್ತರ್ಭಾವ "common.hpp"


ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ op_repeat(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ b;
    GGML_ಬಳಸದ(a);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ op_ಕೂಡಿಸು(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a + b;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ op_sub(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a - b;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ op_mul(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a * b;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ op_div(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) {
    ಹಿಂತಿರುಗಿ a / b;
}

ಶೂನ್ಯ ggml_sycl_ಕೂಡಿಸು(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ಶೂನ್ಯ ggml_sycl_op_ಕೂಡಿಸು_ಕೂಡಿಸು_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_ಕೂಡಿಸು_kernel_supports(ಎಣಿಕೆ ggml_type src0, ಎಣಿಕೆ ggml_type src1, ಎಣಿಕೆ ggml_type dst) {
    ಆದರೆ (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
    ಆದರೆ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

#ಆದರೆ_ಅಂತ್ಯ //GGML_SYCL_BINBCAST_HPP

