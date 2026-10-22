//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_scale_params {
    ସଂରଚନା ggml_tensor src0;   // F32 input tensor
    ସଂରଚନା ggml_tensor dst;    // F32 output tensor
    ଦଶମିକ              scale;  // Scale factor
    ଦଶମିକ              bias;   // Bias (additive offset)
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_scale_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }


    ଦଶମିକ scale = params->scale;
    ଦଶମିକ bias  = params->bias;

    // Total elements across all dimensions
    ଅଚଳ int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ଅଚଳ int64_t elements_per_cacheline = 16;
    ଅଚଳ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    ଯଦି (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    ଯଦି (start_cacheline >= total_cachelines) {
        ଫେରନ୍ତୁ 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    ଯଦି (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");
    __asm__ ଅସ୍ଥିର("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ ଅସ୍ଥିର("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    ପାଇଁ (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(ଦଶମିକ (*)[8]) & dst_data[i])
            : [src] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));

    ଫେରନ୍ତୁ 0;
}
