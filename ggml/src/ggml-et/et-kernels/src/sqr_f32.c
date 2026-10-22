//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
ਬਣਤਰ ggml_et_sqr_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input tensor
    ਬਣਤਰ ggml_tensor dst;   // F32 output tensor
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_sqr_params * params, ਖਾਲੀ * env) {
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
        ਵਾਪਸ -1;  // Invalid pointer
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;  // Unsupported type combination
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ਅਚਲ int64_t ਕੁੱਲ_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ਅਚਲ int64_t elements_ਪ੍ਰਤੀ_cacheline = 16;  // 64 bytes / 4 bytes per float
    ਅਚਲ int64_t ਕੁੱਲ_cachelines       = (ਕੁੱਲ_elements + elements_ਪ੍ਰਤੀ_cacheline - 1) / elements_ਪ੍ਰਤੀ_cacheline;

    ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cachelines + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = thread_id * cl_ਪ੍ਰਤੀ_thread;
    int64_t       cl_ਸਿਰਾ        = cl_ਸ਼ੁਰੂ + cl_ਪ੍ਰਤੀ_thread;
    ਜੇ (cl_ਸਿਰਾ > ਕੁੱਲ_cachelines) {
        cl_ਸਿਰਾ = ਕੁੱਲ_cachelines;
    }

    ਜੇ (cl_ਸ਼ੁਰੂ >= ਕੁੱਲ_cachelines) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ int64_t elem_ਸ਼ੁਰੂ = cl_ਸ਼ੁਰੂ * elements_ਪ੍ਰਤੀ_cacheline;
    int64_t       elem_ਸਿਰਾ   = cl_ਸਿਰਾ * elements_ਪ੍ਰਤੀ_cacheline;
    ਜੇ (elem_ਸਿਰਾ > ਕੁੱਲ_elements) {
        elem_ਸਿਰਾ = ਕੁੱਲ_elements;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr = src0_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ;
    ਦਸ਼ਮਲਵ *       dst_ptr = dst_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ;
    ਅਚਲ int32_t ਗਿਣਤੀ_2   = (int32_t) (elem_ਸਿਰਾ - elem_ਸ਼ੁਰੂ);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    ਲਈ (int32_t i0 = 0; i0 < ਗਿਣਤੀ_2; i0 += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [ਨਤੀਜਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    ਵਾਪਸ 0;
}
