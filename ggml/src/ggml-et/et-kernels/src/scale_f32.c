//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_scale_params {
    काष्ठा ggml_tensor src0;   // F32 input tensor
    काष्ठा ggml_tensor dst;    // F32 output tensor
    भग्न              scale;  // Scale factor
    भग्न              bias;   // Bias (additive offset)
};

पूर्णांक entry_point(काष्ठा ggml_et_scale_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }


    भग्न scale = params->scale;
    भग्न bias  = params->bias;

    // Total elements across all dimensions
    स्थिर int64_t योग_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    स्थिर int64_t elements_per_cacheline = 16;
    स्थिर int64_t योग_cachelines       = (योग_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (योग_cachelines + अंक_threads - 1) / अंक_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    अगर (end_cacheline > योग_cachelines) {
        end_cacheline = योग_cachelines;
    }

    अगर (start_cacheline >= योग_cachelines) {
        वापस 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    अगर (end_elem > योग_elements) {
        end_elem = योग_elements;
    }

    अचिन्हित दीर्घ अस्थायी_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(अस्थायी_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");
    __asm__ अस्थिर("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ अस्थिर("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    क्रम (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(भग्न (*)[8]) & dst_तथ्य[i])
            : [src] "m"(*(स्थिर भग्न (*)[8]) & src0_तथ्य[i])
            : "f10", "f20", "f21");
    }
    __asm__ अस्थिर("mova.m.x %0" ::"r"(अस्थायी_mask));

    वापस 0;
}
