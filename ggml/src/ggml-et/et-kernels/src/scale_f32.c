//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_scale_params {
    بنية ggml_tensor src0;   // F32 input tensor
    بنية ggml_tensor dst;    // F32 output tensor
    عائم              scale;  // Scale factor
    عائم              bias;   // Bias (additive offset)
};

صحيح entry_point(بنية ggml_et_scale_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    𐡄𐡍 (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    𐡄𐡍 (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    𐡄𐡍 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_data = (عائم *) src0->data;
    عائم * dst_data  = (عائم *) dst->data;

    𐡄𐡍 (!src0_data || !dst_data) {
        أرجع -1;
    }


    عائم scale = params->scale;
    عائم bias  = params->bias;

    // Total elements across all dimensions
    ثابت int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ثابت int64_t elements_per_cacheline = 16;
    ثابت int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + num_threads - 1) / num_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    𐡄𐡍 (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    𐡄𐡍 (start_cacheline >= total_cachelines) {
        أرجع 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    𐡄𐡍 (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ متطاير("mova.x.m %0" : "=r"(temp_mask));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");
    __asm__ متطاير("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ متطاير("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    لكل (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(عائم (*)[8]) & dst_data[i])
            : [src] "m"(*(ثابت عائم (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ متطاير("mova.m.x %0" ::"r"(temp_mask));

    أرجع 0;
}
