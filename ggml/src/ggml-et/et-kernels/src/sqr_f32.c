//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
ساختار ggml_et_sqr_params {
    ساختار ggml_tensor src0;  // F32 input tensor
    ساختار ggml_tensor dst;   // F32 output tensor
};

صحیح entry_point(ساختار ggml_et_sqr_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;  // Invalid pointer
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;  // Unsupported type combination
    }

    اعشاری * src0_data = (اعشاری *) src0->data;
    اعشاری * dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ثابت int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ثابت int64_t cl_per_thread = (total_cachelines + رقم_threads - 1) / رقم_threads;
    ثابت int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    اگر (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    اگر (cl_start >= total_cachelines) {
        بازگشت 0;
    }

    ثابت int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    اگر (elem_end > total_elements) {
        elem_end = total_elements;
    }

    ثابت اعشاری * src_ptr = src0_data + elem_start;
    اعشاری *       dst_ptr = dst_data + elem_start;
    ثابت int32_t شمار   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    برای (int32_t i0 = 0; i0 < شمار; i0 += 8) {
        __asm__ فرار(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    بازگشت 0;
}
