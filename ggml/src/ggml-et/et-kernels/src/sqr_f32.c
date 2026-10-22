//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
بنية ggml_et_sqr_params {
    بنية ggml_tensor src0;  // F32 input tensor
    بنية ggml_tensor dst;   // F32 output tensor
};

صحيح entry_point(بنية ggml_et_sqr_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        إرجاع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    إذا (thread_id < 0) {
        إرجاع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        إرجاع -1;  // Invalid pointer
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        إرجاع -1;  // Unsupported type combination
    }

    عائم * src0_data = (عائم *) src0->data;
    عائم * dst_data  = (عائم *) dst->data;

    إذا (!src0_data || !dst_data) {
        إرجاع -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ثابت int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ثابت int64_t cl_per_thread = (total_cachelines + رقم_threads - 1) / رقم_threads;
    ثابت int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    إذا (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    إذا (cl_start >= total_cachelines) {
        إرجاع 0;
    }

    ثابت int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    إذا (elem_end > total_elements) {
        elem_end = total_elements;
    }

    ثابت عائم * src_ptr = src0_data + elem_start;
    عائم *       dst_ptr = dst_data + elem_start;
    ثابت int32_t تعداد_2   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    لكل (int32_t i0 = 0; i0 < تعداد_2; i0 += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    إرجاع 0;
}
