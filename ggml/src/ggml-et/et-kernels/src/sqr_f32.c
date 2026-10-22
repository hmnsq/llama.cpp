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

पूर्णांक entry_point(संरचना ggml_et_sqr_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;  // Invalid pointer
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;  // Unsupported type combination
    }

    दशांश * src0_data = (दशांश *) src0->data;
    दशांश * dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !dst_data) {
        परत -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    स्थिरांक int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    स्थिरांक int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    स्थिरांक int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    स्थिरांक int64_t cl_per_thread = (total_cachelines + अंक_threads - 1) / अंक_threads;
    स्थिरांक int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    जर (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    जर (cl_start >= total_cachelines) {
        परत 0;
    }

    स्थिरांक int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    जर (elem_end > total_elements) {
        elem_end = total_elements;
    }

    स्थिरांक दशांश * src_ptr = src0_data + elem_start;
    दशांश *       dst_ptr = dst_data + elem_start;
    स्थिरांक int32_t मोजणी   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    साठी (int32_t i0 = 0; i0 < मोजणी; i0 += 8) {
        __asm__ अस्थिर(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(दशांश (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(स्थिरांक दशांश (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    परत 0;
}
