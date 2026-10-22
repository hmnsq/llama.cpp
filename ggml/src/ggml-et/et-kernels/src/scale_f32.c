//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_scale_params {
    ഘടന ggml_tensor src0;   // F32 input tensor
    ഘടന ggml_tensor dst;    // F32 output tensor
    ദശാംശം              scale;  // Scale factor
    ദശാംശം              bias;   // Bias (additive offset)
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_scale_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }


    ദശാംശം scale = params->scale;
    ദശാംശം bias  = params->bias;

    // Total elements across all dimensions
    സ്ഥിരം int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    സ്ഥിരം int64_t elements_per_cacheline = 16;
    സ്ഥിരം int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + അക്കം_threads - 1) / അക്കം_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    എങ്കിൽ (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    എങ്കിൽ (start_cacheline >= total_cachelines) {
        തിരികെ 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    എങ്കിൽ (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(temp_mask));
    __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");
    __asm__ അസ്ഥിര("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ അസ്ഥിര("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    വേണ്ടി (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ അസ്ഥിര(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(ദശാംശം (*)[8]) & dst_data[i])
            : [src] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(temp_mask));

    തിരികെ 0;
}
