//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_fill_params {
    ਬਣਤਰ ggml_tensor dst;  // F32 output tensor (contiguous)
    ਦਸ਼ਮਲਵ              c;    // Constant value to fill
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_fill_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * dst = &params->dst;

    ਜੇ (dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਜੇ (!dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ਕੁੱਲ_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    ਜੇ (ਕੁੱਲ_elements == 0) {
        ਵਾਪਸ 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl  = 16;
    ਅਚਲ int64_t ਕੁੱਲ_cl      = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;
    ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cl + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = thread_id * cl_ਪ੍ਰਤੀ_thread;
    int64_t       cl_ਸਿਰਾ        = cl_ਸ਼ੁਰੂ + cl_ਪ੍ਰਤੀ_thread;
    ਜੇ (cl_ਸਿਰਾ > ਕੁੱਲ_cl) {
        cl_ਸਿਰਾ = ਕੁੱਲ_cl;
    }
    ਜੇ (cl_ਸ਼ੁਰੂ >= ਕੁੱਲ_cl) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ int64_t es = cl_ਸ਼ੁਰੂ * elems_ਪ੍ਰਤੀ_cl;
    int64_t       ee = cl_ਸਿਰਾ * elems_ਪ੍ਰਤੀ_cl;
    ਜੇ (ee > ਕੁੱਲ_elements) {
        ee = ਕੁੱਲ_elements;
    }

    // Broadcast constant to all SIMD lanes
    ਦਸ਼ਮਲਵ c = params->c;
    __asm__ ਅਸਥਿਰ("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ਅਚਲ int64_t vec_ਸਿਰਾ = es + ((ee - es) / 8) * 8;
    ਲਈ (; i < vec_ਸਿਰਾ; i += 8) {
        __asm__ ਅਸਥਿਰ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਅੰਕੜੇ[i])::"f10");
    }
    // Scalar tail
    ਲਈ (; i < ee; i++) {
        dst_ਅੰਕੜੇ[i] = c;
    }

    ਵਾਪਸ 0;
}
