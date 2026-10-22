//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_scale_params {
    কাঠামো ggml_tensor src0;   // F32 input tensor
    কাঠামো ggml_tensor dst;    // F32 output tensor
    দশমিক              scale;  // Scale factor
    দশমিক              bias;   // Bias (additive offset)
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_scale_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }


    দশমিক scale = params->scale;
    দশমিক bias  = params->bias;

    // Total elements across all dimensions
    ধ্রুবক int64_t যোগ_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ধ্রুবক int64_t elements_per_cacheline = 16;
    ধ্রুবক int64_t যোগ_cachelines       = (যোগ_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (যোগ_cachelines + অঙ্ক_threads - 1) / অঙ্ক_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    যদি (end_cacheline > যোগ_cachelines) {
        end_cacheline = যোগ_cachelines;
    }

    যদি (start_cacheline >= যোগ_cachelines) {
        ফেরত 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    যদি (end_elem > যোগ_elements) {
        end_elem = যোগ_elements;
    }

    unsigned long অস্থায়ী_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");
    __asm__ অস্থির("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ অস্থির("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    জন্য (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ অস্থির(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(দশমিক (*)[8]) & dst_তথ্য[i])
            : [src] "m"(*(ধ্রুবক দশমিক (*)[8]) & src0_তথ্য[i])
            : "f10", "f20", "f21");
    }
    __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));

    ফেরত 0;
}
