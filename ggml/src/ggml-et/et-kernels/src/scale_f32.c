//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_scale_params {
    संरचना ggml_tensor src0;   // F32 input tensor
    संरचना ggml_tensor dst;    // F32 output tensor
    भिन्न              scale;  // Scale factor
    भिन्न              bias;   // Bias (additive offset)
};

पूर्णाङ्क entry_point(संरचना ggml_et_scale_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        फर्काऊ -1;
    }


    भिन्न scale = params->scale;
    भिन्न bias  = params->bias;

    // Total elements across all dimensions
    अचर int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    अचर int64_t elements_per_cacheline = 16;
    अचर int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + अङ्क_threads - 1) / अङ्क_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    यदि (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    यदि (start_cacheline >= total_cachelines) {
        फर्काऊ 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    यदि (end_elem > total_elements) {
        end_elem = total_elements;
    }

    अचिन्हित दीर्घ temp_mask;
    __asm__ चञ्चल("mova.x.m %0" : "=r"(temp_mask));
    __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");
    __asm__ चञ्चल("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ चञ्चल("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    लागि (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ चञ्चल(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(भिन्न (*)[8]) & dst_data[i])
            : [src] "m"(*(अचर भिन्न (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ चञ्चल("mova.m.x %0" ::"r"(temp_mask));

    फर्काऊ 0;
}
