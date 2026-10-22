//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_scale_params {
    கட்டமைப்பு ggml_tensor src0;   // F32 input tensor
    கட்டமைப்பு ggml_tensor dst;    // F32 output tensor
    தசம              scale;  // Scale factor
    தசம              bias;   // Bias (additive offset)
};

முழு entry_point(கட்டமைப்பு ggml_et_scale_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }


    தசம scale = params->scale;
    தசம bias  = params->bias;

    // Total elements across all dimensions
    மாறிலி int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    மாறிலி int64_t elements_per_cacheline = 16;
    மாறிலி int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + இலக்கம்_threads - 1) / இலக்கம்_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    எனில் (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    எனில் (start_cacheline >= total_cachelines) {
        திரும்பு 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    எனில் (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ நிலையற்ற("mova.x.m %0" : "=r"(temp_mask));
    __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");
    __asm__ நிலையற்ற("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ நிலையற்ற("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    க்கு (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ நிலையற்ற(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(தசம (*)[8]) & dst_data[i])
            : [src] "m"(*(மாறிலி தசம (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ நிலையற்ற("mova.m.x %0" ::"r"(temp_mask));

    திரும்பு 0;
}
