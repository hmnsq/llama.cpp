//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
কাঠামো ggml_et_sqr_params {
    কাঠামো ggml_tensor src0;  // F32 input tensor
    কাঠামো ggml_tensor dst;   // F32 output tensor
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_sqr_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;  // Invalid pointer
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;  // Unsupported type combination
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ধ্রুবক int64_t যোগ_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ধ্রুবক int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ধ্রুবক int64_t যোগ_cachelines       = (যোগ_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ধ্রুবক int64_t cl_per_thread = (যোগ_cachelines + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    যদি (cl_end > যোগ_cachelines) {
        cl_end = যোগ_cachelines;
    }

    যদি (cl_start >= যোগ_cachelines) {
        ফেরত 0;
    }

    ধ্রুবক int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    যদি (elem_end > যোগ_elements) {
        elem_end = যোগ_elements;
    }

    ধ্রুবক দশমিক * src_ptr = src0_তথ্য + elem_start;
    দশমিক *       dst_ptr = dst_তথ্য + elem_start;
    ধ্রুবক int32_t গুনতি   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    জন্য (int32_t i0 = 0; i0 < গুনতি; i0 += 8) {
        __asm__ অস্থির(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [ফলাফল] "=m"(*(দশমিক (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ধ্রুবক দশমিক (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    ফেরত 0;
}
