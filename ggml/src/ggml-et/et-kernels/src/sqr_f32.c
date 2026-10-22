//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
sa_nracanaa ggml_et_sqr_params {
    sa_nracanaa ggml_tensor src0;  // F32 input tensor
    sa_nracanaa ggml_tensor dst;   // F32 output tensor
};

පූර්ණ_සංඛ්‍යා entry_point(sa_nracanaa ggml_et_sqr_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    නම් (!kernel_env) {
        ආපසු -1;
    }

    පූර්ණ_සංඛ්‍යා thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    පූර්ණ_සංඛ්‍යා num_threads = get_num_threads(kernel_env->shire_mask);

    නම් (thread_id < 0) {
        ආපසු 0;
    }

    නම් (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ආපසු -1;  // Invalid pointer
    }

    sa_nracanaa ggml_tensor * src0 = &params->src0;
    sa_nracanaa ggml_tensor * dst  = &params->dst;

    නම් (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ආපසු -1;  // Unsupported type combination
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    නම් (!src0_data || !dst_data) {
        ආපසු -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    const int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    const int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    const int64_t cl_per_thread = (total_cachelines + num_threads - 1) / num_threads;
    const int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    නම් (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    නම් (cl_start >= total_cachelines) {
        ආපසු 0;
    }

    const int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    නම් (elem_end > total_elements) {
        elem_end = total_elements;
    }

    const float * src_ptr = src0_data + elem_start;
    float *       dst_ptr = dst_data + elem_start;
    const int32_t count   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    සඳහා (int32_t i0 = 0; i0 < count; i0 += 8) {
        __asm__ ca_ncala(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    ආපසු 0;
}
