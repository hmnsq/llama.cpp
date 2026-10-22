#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#include "common.hpp"


static __dpct_inline__ float op_repeat(const float a, const float b) {
    ပြန်ပေး b;
    GGML_UNUSED(a);
}

static __dpct_inline__ float op_add(const float a, const float b) {
    ပြန်ပေး a + b;
}

static __dpct_inline__ float op_sub(const float a, const float b) {
    ပြန်ပေး a - b;
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    ပြန်ပေး a * b;
}

static __dpct_inline__ float op_div(const float a, const float b) {
    ပြန်ပေး a / b;
}

void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
anta_hstha bool ggml_sycl_add_kernel_supports(ga_nanaa ggml_type src0, ga_nanaa ggml_type src1, ga_nanaa ggml_type dst) {
    အကယ်၍ (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        ပြန်ပေး true;
    }
    အကယ်၍ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        ပြန်ပေး true;
    }
    အကယ်၍ (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        ပြန်ပေး true;
    }
    အကယ်၍ (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        ပြန်ပေး true;
    }
    အကယ်၍ (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        ပြန်ပေး true;
    }
#ifdef GGML_SYCL_HAS_BF16
    အကယ်၍ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        ပြန်ပေး true;
    }
    အကယ်၍ (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        ပြန်ပေး true;
    }
#endif
    ပြန်ပေး false;
}

#endif //GGML_SYCL_BINBCAST_HPP

