#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#شامل "common.hpp"


ایستا __dpct_inline__ اعشاری op_repeat(ثابت اعشاری a, ثابت اعشاری b) {
    بازگشت b;
    GGML_UNUSED(a);
}

ایستا __dpct_inline__ اعشاری op_add(ثابت اعشاری a, ثابت اعشاری b) {
    بازگشت a + b;
}

ایستا __dpct_inline__ اعشاری op_sub(ثابت اعشاری a, ثابت اعشاری b) {
    بازگشت a - b;
}

ایستا __dpct_inline__ اعشاری op_mul(ثابت اعشاری a, ثابت اعشاری b) {
    بازگشت a * b;
}

ایستا __dpct_inline__ اعشاری op_div(ثابت اعشاری a, ثابت اعشاری b) {
    بازگشت a / b;
}

تهی ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

تهی ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

تهی ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

تهی ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

تهی ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

تهی ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
درونخطی منطقی ggml_sycl_add_kernel_supports(شمارش ggml_type src0, شمارش ggml_type src1, شمارش ggml_type dst) {
    اگر (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        بازگشت درست;
    }
    اگر (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        بازگشت درست;
    }
    اگر (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        بازگشت درست;
    }
    اگر (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        بازگشت درست;
    }
    اگر (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        بازگشت درست;
    }
#ifdef GGML_SYCL_HAS_BF16
    اگر (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        بازگشت درست;
    }
    اگر (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        بازگشت درست;
    }
#endif
    بازگشت نادرست;
}

#endif //GGML_SYCL_BINBCAST_HPP

