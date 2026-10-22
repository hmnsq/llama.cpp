//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_scale_params {
    ساختار ggml_tensor src0;   // F32 input tensor
    ساختار ggml_tensor dst;    // F32 output tensor
    اعشاری              scale;  // Scale factor
    اعشاری              bias;   // Bias (additive offset)
};

صحیح entry_point(ساختار ggml_et_scale_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    اعشاری * src0_data = (اعشاری *) src0->data;
    اعشاری * dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;
    }


    اعشاری scale = params->scale;
    اعشاری bias  = params->bias;

    // Total elements across all dimensions
    ثابت int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ثابت int64_t elements_per_cacheline = 16;
    ثابت int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t cachelines_per_thread = (total_cachelines + رقم_threads - 1) / رقم_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    اگر (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    اگر (start_cacheline >= total_cachelines) {
        بازگشت 0;
    }

    int64_t start_elem = start_cacheline * elements_per_cacheline;
    int64_t end_elem   = end_cacheline * elements_per_cacheline;
    اگر (end_elem > total_elements) {
        end_elem = total_elements;
    }

    unsigned long temp_mask;
    __asm__ فرار("mova.x.m %0" : "=r"(temp_mask));
    __asm__ فرار("mov.m.x m0, x0, 0xFF");
    __asm__ فرار("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ فرار("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    برای (int64_t i = start_elem; i < end_elem; i += 8) {
        __asm__ فرار(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_out] "=m"(*(اعشاری (*)[8]) & dst_data[i])
            : [src] "m"(*(ثابت اعشاری (*)[8]) & src0_data[i])
            : "f10", "f20", "f21");
    }
    __asm__ فرار("mova.m.x %0" ::"r"(temp_mask));

    بازگشت 0;
}
