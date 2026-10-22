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
    दशांश              scale;  // Scale factor
    दशांश              bias;   // Bias (additive offset)
};

पूर्णांक entry_point(संरचना ggml_et_scale_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    दशांश * src0_data = (दशांश *) src0->data;
    दशांश * dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !dst_data) {
        परत -1;
    }


    दशांश scale = params->scale;
    दशांश bias  = params->bias;

    // Total elements across all dimensions
    स्थिरांक int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    स्थिरांक int64_t elements_per_cacheline = 16;
    स्थिरांक int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + अंक_threads - 1) / अंक_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    जर (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    जर (start_cacheline >= total_cachelines) {
        परत 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    जर (end_elem > total_elements) {
        end_elem = total_elements;
    }

    अचिन्हित दीर्घ temp_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");
    __asm__ अस्थिर("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ अस्थिर("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    साठी (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(दशांश (*)[8]) & dst_data[i])
            : [src] "m"(*(स्थिरांक दशांश (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));

    परत 0;
}
