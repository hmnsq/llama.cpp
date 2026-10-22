#اگر_غیر_تعریف_شدہ GGML_SYCL_BINBCAST_HPP
#تعریف_کرو GGML_SYCL_BINBCAST_HPP
#شامل "common.hpp"


ساکن __dpct_inline__ اعشاری op_repeat(ثابت اعشاری a, ثابت اعشاری b) {
    واپس b;
    GGML_غیر_مستعمل(a);
}

ساکن __dpct_inline__ اعشاری op_جوڑو(ثابت اعشاری a, ثابت اعشاری b) {
    واپس a + b;
}

ساکن __dpct_inline__ اعشاری op_sub(ثابت اعشاری a, ثابت اعشاری b) {
    واپس a - b;
}

ساکن __dpct_inline__ اعشاری op_mul(ثابت اعشاری a, ثابت اعشاری b) {
    واپس a * b;
}

ساکن __dpct_inline__ اعشاری op_div(ثابت اعشاری a, ثابت اعشاری b) {
    واپس a / b;
}

خالی ggml_sycl_جوڑو(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

خالی ggml_sycl_op_جوڑو_جوڑو_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
اندرونی منطقی ggml_sycl_جوڑو_kernel_supports(شمار ggml_type src0, شمار ggml_type src1, شمار ggml_type dst) {
    اگر (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        واپس سچ;
    }
    اگر (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        واپس سچ;
    }
    اگر (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        واپس سچ;
    }
    اگر (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        واپس سچ;
    }
    اگر (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        واپس سچ;
    }
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
    اگر (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        واپس سچ;
    }
    اگر (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        واپس سچ;
    }
#اگر_ختم
    واپس جھوٹ;
}

#اگر_ختم //GGML_SYCL_BINBCAST_HPP

