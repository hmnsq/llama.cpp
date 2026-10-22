//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
সংরচনা ggml_et_sqr_params {
    সংরচনা ggml_tensor src0;  // F32 input tensor
    সংরচনা ggml_tensor dst;   // F32 output tensor
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_sqr_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;  // Invalid pointer
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;  // Unsupported type combination
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        প্রত্যাগম -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ধ্রুব int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ধ্রুব int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ধ্রুব int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ধ্রুব int64_t cl_per_thread = (total_cachelines + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুব int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    যদি (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    যদি (cl_start >= total_cachelines) {
        প্রত্যাগম 0;
    }

    ধ্রুব int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    যদি (elem_end > total_elements) {
        elem_end = total_elements;
    }

    ধ্রুব ভিন্ন * src_ptr = src0_data + elem_start;
    ভিন্ন *       dst_ptr = dst_data + elem_start;
    ধ্রুব int32_t গুনতি   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    প্রতি (int32_t i0 = 0; i0 < গুনতি; i0 += 8) {
        __asm__ চঞ্চল(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(ভিন্ন (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    প্রত্যাগম 0;
}
