//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
संरचना ggml_et_sqr_params {
    संरचना ggml_tensor src0;  // F32 input tensor
    संरचना ggml_tensor dst;   // F32 output tensor
};

पूर्णाङ्क entry_point(संरचना ggml_et_sqr_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;  // Invalid pointer
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;  // Unsupported type combination
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        प्रत्यागम -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ध्रुव int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ध्रुव int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ध्रुव int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ध्रुव int64_t cl_per_thread = (total_cachelines + अंक_threads - 1) / अंक_threads;
    ध्रुव int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    यदि (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    यदि (cl_start >= total_cachelines) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    यदि (elem_end > total_elements) {
        elem_end = total_elements;
    }

    ध्रुव भिन्न * src_ptr = src0_data + elem_start;
    भिन्न *       dst_ptr = dst_data + elem_start;
    ध्रुव int32_t गिनती   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    प्रति (int32_t i0 = 0; i0 < गिनती; i0 += 8) {
        __asm__ चञ्चल(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(भिन्न (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    प्रत्यागम 0;
}
