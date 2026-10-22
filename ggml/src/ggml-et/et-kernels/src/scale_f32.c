//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

sa_nracanaa ggml_et_scale_params {
    sa_nracanaa ggml_tensor src0;   // F32 input tensor
    sa_nracanaa ggml_tensor dst;    // F32 output tensor
    float              scale;  // Scale factor
    float              bias;   // Bias (additive offset)
};

puur_naa_nka entry_point(sa_nracanaa ggml_et_scale_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ᬬᬾᬦ᭄ (!kernel_env) {
        ᬯᬮᬶ -1;
    }

    puur_naa_nka thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    puur_naa_nka num_threads = get_num_threads(kernel_env->shire_mask);

    ᬬᬾᬦ᭄ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ᬯᬮᬶ -1;
    }

    sa_nracanaa ggml_tensor * src0 = &params->src0;
    sa_nracanaa ggml_tensor * dst  = &params->dst;

    ᬬᬾᬦ᭄ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ᬯᬮᬶ -1;
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    ᬬᬾᬦ᭄ (!src0_data || !dst_data) {
        ᬯᬮᬶ -1;
    }


    float scale = params->scale;
    float bias  = params->bias;

    // Total elements across all dimensions
    const int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    const int64_t elements_per_cacheline = 16;
    const int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + num_threads - 1) / num_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    ᬬᬾᬦ᭄ (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    ᬬᬾᬦ᭄ (start_cacheline >= total_cachelines) {
        ᬯᬮᬶ 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    ᬬᬾᬦ᭄ (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ ca_ncala("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ca_ncala("mov.m.x m0, x0, 0xFF");
    __asm__ ca_ncala("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ ca_ncala("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    for (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ ca_ncala(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(float (*)[8]) & dst_data[i])
            : [src] "m"(*(const float (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ ca_ncala("mova.m.x %0" ::"r"(temp_mask));

    ᬯᬮᬶ 0;
}
