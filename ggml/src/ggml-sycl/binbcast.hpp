#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#അന്തര്ഭാവ "common.hpp"


സ്ഥിതം __dpct_inline__ ദശാംശം op_repeat(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) {
    തിരികെ b;
    GGML_UNUSED(a);
}

സ്ഥിതം __dpct_inline__ ദശാംശം op_add(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) {
    തിരികെ a + b;
}

സ്ഥിതം __dpct_inline__ ദശാംശം op_sub(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) {
    തിരികെ a - b;
}

സ്ഥിതം __dpct_inline__ ദശാംശം op_mul(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) {
    തിരികെ a * b;
}

സ്ഥിതം __dpct_inline__ ദശാംശം op_div(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) {
    തിരികെ a / b;
}

ശൂന്യം ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ശൂന്യം ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ശൂന്യം ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ശൂന്യം ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ശൂന്യം ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

ശൂന്യം ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
വരിയിൽ തർക്കം ggml_sycl_add_kernel_supports(എണ്ണൽ ggml_type src0, എണ്ണൽ ggml_type src1, എണ്ണൽ ggml_type dst) {
    എങ്കിൽ (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        തിരികെ സത്യം;
    }
    എങ്കിൽ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        തിരികെ സത്യം;
    }
    എങ്കിൽ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        തിരികെ സത്യം;
    }
    എങ്കിൽ (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        തിരികെ സത്യം;
    }
    എങ്കിൽ (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        തിരികെ സത്യം;
    }
#ifdef GGML_SYCL_HAS_BF16
    എങ്കിൽ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        തിരികെ സത്യം;
    }
    എങ്കിൽ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        തിരികെ സത്യം;
    }
#endif
    തിരികെ അസത്യം;
}

#endif //GGML_SYCL_BINBCAST_HPP

