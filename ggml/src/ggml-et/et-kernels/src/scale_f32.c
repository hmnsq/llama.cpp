//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>

সংরচনা ggml_et_scale_params {
    সংরচনা ggml_tensor src0;   // F32 input tensor
    সংরচনা ggml_tensor dst;    // F32 output tensor
    ভিন্ন              scale;  // Scale factor
    ভিন্ন              bias;   // Bias (additive offset)
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_scale_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        ঘূৰাই -1;
    }


    ভিন্ন scale = params->scale;
    ভিন্ন bias  = params->bias;

    // Total elements across all dimensions
    ধ্রুৱ int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ধ্রুৱ int64_t elements_per_cacheline = 16;
    ধ্রুৱ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + অঙ্ক_threads - 1) / অঙ্ক_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    যদি (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    যদি (start_cacheline >= total_cachelines) {
        ঘূৰাই 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    যদি (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ চঞ্চল("mova.x.m %0" : "=r"(temp_mask));
    __asm__ চঞ্চল("mov.m.x m0, x0, 0xFF");
    __asm__ চঞ্চল("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ চঞ্চল("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    বাবে (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(ভিন্ন (*)[8]) & dst_data[i])
            : [src] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ চঞ্চল("mova.m.x %0" ::"r"(temp_mask));

    ঘূৰাই 0;
}
