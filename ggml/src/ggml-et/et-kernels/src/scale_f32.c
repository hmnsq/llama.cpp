//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_scale_params {
    ಸಂರಚನಾ ggml_tensor src0;   // F32 input tensor
    ಸಂರಚನಾ ggml_tensor dst;    // F32 output tensor
    ಭಿನ್ನ              scale;  // Scale factor
    ಭಿನ್ನ              bias;   // Bias (additive offset)
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_scale_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }


    ಭಿನ್ನ scale = params->scale;
    ಭಿನ್ನ bias  = params->bias;

    // Total elements across all dimensions
    ಧ್ರುವ int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ಧ್ರುವ int64_t elements_per_cacheline = 16;
    ಧ್ರುವ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    ಯದಿ (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    ಯದಿ (start_cacheline >= total_cachelines) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    ಯದಿ (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");
    __asm__ ಚಞ್ಚಲ("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ ಚಞ್ಚಲ("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    ಪ್ರತಿ (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ ಚಞ್ಚಲ(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_data[i])
            : [src] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(temp_mask));

    ಪ್ರತ್ಯಾಗಮ 0;
}
