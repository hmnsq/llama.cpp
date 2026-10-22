#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#অন্তর্ভাৱ "common.hpp"


স্থির __dpct_inline__ ভিন্ন op_repeat(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) {
    ঘূৰাই b;
    GGML_UNUSED(a);
}

স্থির __dpct_inline__ ভিন্ন op_add(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) {
    ঘূৰাই a + b;
}

স্থির __dpct_inline__ ভিন্ন op_sub(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) {
    ঘূৰাই a - b;
}

স্থির __dpct_inline__ ভিন্ন op_mul(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) {
    ঘূৰাই a * b;
}

স্থির __dpct_inline__ ভিন্ন op_div(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) {
    ঘূৰাই a / b;
}

শূন্য ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

শূন্য ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
অন্তঃস্থ তার্কিক ggml_sycl_add_kernel_supports(গণনা ggml_type src0, গণনা ggml_type src1, গণনা ggml_type dst) {
    যদি (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        ঘূৰাই সঁচা;
    }
    যদি (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        ঘূৰাই সঁচা;
    }
    যদি (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        ঘূৰাই সঁচা;
    }
    যদি (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        ঘূৰাই সঁচা;
    }
    যদি (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        ঘূৰাই সঁচা;
    }
#ifdef GGML_SYCL_HAS_BF16
    যদি (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        ঘূৰাই সঁচা;
    }
    যদি (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        ঘূৰাই সঁচা;
    }
#endif
    ঘূৰাই মিছা;
}

#endif //GGML_SYCL_BINBCAST_HPP

