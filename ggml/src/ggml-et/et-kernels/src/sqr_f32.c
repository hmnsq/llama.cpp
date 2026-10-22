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

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_sqr_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

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

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ध्रुव int64_t योग_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ध्रुव int64_t elements_प्रत्येकम्_cacheline = 16;  // 64 bytes / 4 bytes per float
    ध्रुव int64_t योग_cachelines       = (योग_elements + elements_प्रत्येकम्_cacheline - 1) / elements_प्रत्येकम्_cacheline;

    ध्रुव int64_t cl_प्रत्येकम्_thread = (योग_cachelines + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
    ध्रुव int64_t cl_start      = thread_id * cl_प्रत्येकम्_thread;
    int64_t       cl_अन्तः        = cl_start + cl_प्रत्येकम्_thread;
    यदि (cl_अन्तः > योग_cachelines) {
        cl_अन्तः = योग_cachelines;
    }

    यदि (cl_start >= योग_cachelines) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t elem_start = cl_start * elements_प्रत्येकम्_cacheline;
    int64_t       elem_अन्तः   = cl_अन्तः * elements_प्रत्येकम्_cacheline;
    यदि (elem_अन्तः > योग_elements) {
        elem_अन्तः = योग_elements;
    }

    ध्रुव भिन्न * src_ptr = src0_दत्त + elem_start;
    भिन्न *       dst_ptr = dst_दत्त + elem_start;
    ध्रुव int32_t गणनम्   = (int32_t) (elem_अन्तः - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    प्रति (int32_t i0 = 0; i0 < गणनम्; i0 += 8) {
        __asm__ चञ्चल(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [परिणाम] "=m"(*(भिन्न (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    प्रत्यागम 0;
}
