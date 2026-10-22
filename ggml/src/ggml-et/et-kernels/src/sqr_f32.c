//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
સંરચના ggml_et_sqr_params {
    સંરચના ggml_tensor src0;  // F32 input tensor
    સંરચના ggml_tensor dst;   // F32 output tensor
};

પૂર્ણાંક entry_point(સંરચના ggml_et_sqr_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;  // Invalid pointer
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;  // Unsupported type combination
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    અચલ int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    અચલ int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    અચલ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    અચલ int64_t cl_per_thread = (total_cachelines + અંક_threads - 1) / અંક_threads;
    અચલ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    જો (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    જો (cl_start >= total_cachelines) {
        પાછા 0;
    }

    અચલ int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    જો (elem_end > total_elements) {
        elem_end = total_elements;
    }

    અચલ દશાંશ * src_ptr = src0_data + elem_start;
    દશાંશ *       dst_ptr = dst_data + elem_start;
    અચલ int32_t ગણતરી   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    માટે (int32_t i0 = 0; i0 < ગણતરી; i0 += 8) {
        __asm__ અસ્થિર(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(દશાંશ (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    પાછા 0;
}
