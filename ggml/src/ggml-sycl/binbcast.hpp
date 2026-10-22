#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#ଅନ୍ତର୍ଭାଵ "common.hpp"


ସ୍ଥିର __dpct_inline__ ଦଶମିକ op_repeat(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) {
    ଫେରନ୍ତୁ b;
    GGML_UNUSED(a);
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ op_add(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) {
    ଫେରନ୍ତୁ a + b;
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ op_sub(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) {
    ଫେରନ୍ତୁ a - b;
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ op_mul(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) {
    ଫେରନ୍ତୁ a * b;
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ op_div(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) {
    ଫେରନ୍ତୁ a / b;
}

ଶୂନ୍ୟ ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ଶୂନ୍ୟ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ଶୂନ୍ୟ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ଶୂନ୍ୟ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ଶୂନ୍ୟ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ଶୂନ୍ୟ ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_add_kernel_supports(ଗଣନା ggml_type src0, ଗଣନା ggml_type src1, ଗଣନା ggml_type dst) {
    ଯଦି (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଯଦି (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଯଦି (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଯଦି (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଯଦି (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
#ifdef GGML_SYCL_HAS_BF16
    ଯଦି (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଯଦି (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
#endif
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

#endif //GGML_SYCL_BINBCAST_HPP

