#నిర్వచితం_కాకపోతే GGML_SYCL_BINBCAST_HPP
#నిర్వచించు GGML_SYCL_BINBCAST_HPP
#అన్తర్భావ "common.hpp"


స్థిర __dpct_inline__ దశాంశం op_repeat(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) {
    తిరిగి b;
    GGML_వాడని(a);
}

స్థిర __dpct_inline__ దశాంశం op_కూడు(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) {
    తిరిగి a + b;
}

స్థిర __dpct_inline__ దశాంశం op_sub(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) {
    తిరిగి a - b;
}

స్థిర __dpct_inline__ దశాంశం op_mul(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) {
    తిరిగి a * b;
}

స్థిర __dpct_inline__ దశాంశం op_div(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) {
    తిరిగి a / b;
}

శూన్య ggml_sycl_కూడు(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

శూన్య ggml_sycl_op_కూడు_కూడు_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
అంతఃవరుస తార్కిక ggml_sycl_కూడు_kernel_supports(గణన ggml_type src0, గణన ggml_type src1, గణన ggml_type dst) {
    ఒకవేళ (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        తిరిగి నిజం;
    }
    ఒకవేళ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        తిరిగి నిజం;
    }
    ఒకవేళ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        తిరిగి నిజం;
    }
    ఒకవేళ (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        తిరిగి నిజం;
    }
    ఒకవేళ (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        తిరిగి నిజం;
    }
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
    ఒకవేళ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        తిరిగి నిజం;
    }
    ఒకవేళ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        తిరిగి నిజం;
    }
#అయితే_ముగింపు
    తిరిగి అబద్ధం;
}

#అయితే_ముగింపు //GGML_SYCL_BINBCAST_HPP

