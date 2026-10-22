//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_scale_params {
    ਬਣਤਰ ggml_tensor src0;   // F32 input tensor
    ਬਣਤਰ ggml_tensor dst;    // F32 output tensor
    ਦਸ਼ਮਲਵ              scale;  // Scale factor
    ਦਸ਼ਮਲਵ              bias;   // Bias (additive offset)
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_scale_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }


    ਦਸ਼ਮਲਵ scale = params->scale;
    ਦਸ਼ਮਲਵ bias  = params->bias;

    // Total elements across all dimensions
    ਅਚਲ int64_t ਕੁੱਲ_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ਅਚਲ int64_t elements_ਪ੍ਰਤੀ_cacheline = 16;
    ਅਚਲ int64_t ਕੁੱਲ_cachelines       = (ਕੁੱਲ_elements + elements_ਪ੍ਰਤੀ_cacheline - 1) / elements_ਪ੍ਰਤੀ_cacheline;

    int64_t cachelines_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cachelines + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    int64_t ਸ਼ੁਰੂ_cacheline       = thread_id * cachelines_ਪ੍ਰਤੀ_thread;
    int64_t ਸਿਰਾ_cacheline         = ਸ਼ੁਰੂ_cacheline + cachelines_ਪ੍ਰਤੀ_thread;

    ਜੇ (ਸਿਰਾ_cacheline > ਕੁੱਲ_cachelines) {
        ਸਿਰਾ_cacheline = ਕੁੱਲ_cachelines;
    }

    ਜੇ (ਸ਼ੁਰੂ_cacheline >= ਕੁੱਲ_cachelines) {
        ਵਾਪਸ 0;
    }

    int64_t ਸ਼ੁਰੂ_elem = ਸ਼ੁਰੂ_cacheline * elements_ਪ੍ਰਤੀ_cacheline;
    int64_t ਸਿਰਾ_elem   = ਸਿਰਾ_cacheline * elements_ਪ੍ਰਤੀ_cacheline;
    ਜੇ (ਸਿਰਾ_elem > ਕੁੱਲ_elements) {
        ਸਿਰਾ_elem = ਕੁੱਲ_elements;
    }

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");
    __asm__ ਅਸਥਿਰ("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ ਅਸਥਿਰ("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    ਲਈ (int64_t i = ਸ਼ੁਰੂ_elem; i < ਸਿਰਾ_elem; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਅੰਕੜੇ[i])
            : [src] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src0_ਅੰਕੜੇ[i])
            : "f10", "f20", "f21");
    }
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    ਵਾਪਸ 0;
}
