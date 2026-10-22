//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_scale_params {
    સંરચના ggml_tensor src0;   // F32 input tensor
    સંરચના ggml_tensor dst;    // F32 output tensor
    દશાંશ              scale;  // Scale factor
    દશાંશ              bias;   // Bias (additive offset)
};

પૂર્ણાંક entry_point(સંરચના ggml_et_scale_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }


    દશાંશ scale = params->scale;
    દશાંશ bias  = params->bias;

    // Total elements across all dimensions
    અચલ int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    અચલ int64_t elements_per_cacheline = 16;
    અચલ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + અંક_threads - 1) / અંક_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    જો (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    જો (start_cacheline >= total_cachelines) {
        પાછા 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    જો (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");
    __asm__ અસ્થિર("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ અસ્થિર("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    માટે (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(દશાંશ (*)[8]) & dst_data[i])
            : [src] "m"(*(અચલ દશાંશ (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));

    પાછા 0;
}
