#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#அந்தர்பாவ "common.hpp"


நிலையான __dpct_inline__ தசம op_repeat(மாறிலி தசம a, மாறிலி தசம b) {
    திரும்பு b;
    GGML_UNUSED(a);
}

நிலையான __dpct_inline__ தசம op_add(மாறிலி தசம a, மாறிலி தசம b) {
    திரும்பு a + b;
}

நிலையான __dpct_inline__ தசம op_sub(மாறிலி தசம a, மாறிலி தசம b) {
    திரும்பு a - b;
}

நிலையான __dpct_inline__ தசம op_mul(மாறிலி தசம a, மாறிலி தசம b) {
    திரும்பு a * b;
}

நிலையான __dpct_inline__ தசம op_div(மாறிலி தசம a, மாறிலி தசம b) {
    திரும்பு a / b;
}

வெறுமை ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

வெறுமை ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

வெறுமை ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

வெறுமை ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

வெறுமை ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

வெறுமை ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
உள்வரி தருக்க ggml_sycl_add_kernel_supports(எண்ணிக்கை ggml_type src0, எண்ணிக்கை ggml_type src1, எண்ணிக்கை ggml_type dst) {
    எனில் (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        திரும்பு உண்மை;
    }
    எனில் (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        திரும்பு உண்மை;
    }
    எனில் (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        திரும்பு உண்மை;
    }
    எனில் (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        திரும்பு உண்மை;
    }
    எனில் (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        திரும்பு உண்மை;
    }
#ifdef GGML_SYCL_HAS_BF16
    எனில் (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        திரும்பு உண்மை;
    }
    எனில் (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        திரும்பு உண்மை;
    }
#endif
    திரும்பு பொய்;
}

#endif //GGML_SYCL_BINBCAST_HPP

