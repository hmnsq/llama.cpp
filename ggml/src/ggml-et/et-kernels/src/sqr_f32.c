//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
काष्ठा ggml_et_sqr_params {
    काष्ठा ggml_tensor src0;  // F32 input tensor
    काष्ठा ggml_tensor dst;   // F32 output tensor
};

पूर्णांक entry_point(काष्ठा ggml_et_sqr_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;  // Invalid pointer
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;  // Unsupported type combination
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    स्थिर int64_t योग_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    स्थिर int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    स्थिर int64_t योग_cachelines       = (योग_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    स्थिर int64_t cl_per_thread = (योग_cachelines + अंक_threads - 1) / अंक_threads;
    स्थिर int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    अगर (cl_end > योग_cachelines) {
        cl_end = योग_cachelines;
    }

    अगर (cl_start >= योग_cachelines) {
        वापस 0;
    }

    स्थिर int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    अगर (elem_end > योग_elements) {
        elem_end = योग_elements;
    }

    स्थिर भग्न * src_ptr = src0_तथ्य + elem_start;
    भग्न *       dst_ptr = dst_तथ्य + elem_start;
    स्थिर int32_t गिनती   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    क्रम (int32_t i0 = 0; i0 < गिनती; i0 += 8) {
        __asm__ अस्थिर(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [परिणाम] "=m"(*(भग्न (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    वापस 0;
}
