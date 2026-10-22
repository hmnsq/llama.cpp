#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#અન્તર્ભાવ "common.hpp"


સ્થિર __dpct_inline__ દશાંશ op_repeat(અચલ દશાંશ a, અચલ દશાંશ b) {
    પાછા b;
    GGML_UNUSED(a);
}

સ્થિર __dpct_inline__ દશાંશ op_add(અચલ દશાંશ a, અચલ દશાંશ b) {
    પાછા a + b;
}

સ્થિર __dpct_inline__ દશાંશ op_sub(અચલ દશાંશ a, અચલ દશાંશ b) {
    પાછા a - b;
}

સ્થિર __dpct_inline__ દશાંશ op_mul(અચલ દશાંશ a, અચલ દશાંશ b) {
    પાછા a * b;
}

સ્થિર __dpct_inline__ દશાંશ op_div(અચલ દશાંશ a, અચલ દશાંશ b) {
    પાછા a / b;
}

શૂન્ય ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

શૂન્ય ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

શૂન્ય ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

શૂન્ય ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

શૂન્ય ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

શૂન્ય ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
અંતઃરેખ તાર્કિક ggml_sycl_add_kernel_supports(ગણના ggml_type src0, ગણના ggml_type src1, ગણના ggml_type dst) {
    જો (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        પાછા સાચું;
    }
    જો (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        પાછા સાચું;
    }
    જો (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        પાછા સાચું;
    }
    જો (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        પાછા સાચું;
    }
    જો (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        પાછા સાચું;
    }
#ifdef GGML_SYCL_HAS_BF16
    જો (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        પાછા સાચું;
    }
    જો (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        પાછા સાચું;
    }
#endif
    પાછા ખોટું;
}

#endif //GGML_SYCL_BINBCAST_HPP

