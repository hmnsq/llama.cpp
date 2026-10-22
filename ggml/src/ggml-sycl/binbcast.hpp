#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#ضمن "common.hpp"


ساكن __dpct_inline__ عائم op_repeat(ثابت عائم a, ثابت عائم b) {
    أرجع b;
    GGML_UNUSED(a);
}

ساكن __dpct_inline__ عائم op_add(ثابت عائم a, ثابت عائم b) {
    أرجع a + b;
}

ساكن __dpct_inline__ عائم op_sub(ثابت عائم a, ثابت عائم b) {
    أرجع a - b;
}

ساكن __dpct_inline__ عائم op_mul(ثابت عائم a, ثابت عائم b) {
    أرجع a * b;
}

ساكن __dpct_inline__ عائم op_div(ثابت عائم a, ثابت عائم b) {
    أرجع a / b;
}

فراغ ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

فراغ ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
مضمن منطقي ggml_sycl_add_kernel_supports(تعداد ggml_type src0, تعداد ggml_type src1, تعداد ggml_type dst) {
    𐤀𐤌 (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        أرجع true;
    }
    𐤀𐤌 (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        أرجع true;
    }
    𐤀𐤌 (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        أرجع true;
    }
    𐤀𐤌 (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        أرجع true;
    }
    𐤀𐤌 (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        أرجع true;
    }
#ifdef GGML_SYCL_HAS_BF16
    𐤀𐤌 (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        أرجع true;
    }
    𐤀𐤌 (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        أرجع true;
    }
#endif
    أرجع خطأ;
}

#endif //GGML_SYCL_BINBCAST_HPP

