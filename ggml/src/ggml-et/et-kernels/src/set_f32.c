//******************************************************************************
// SET F32 Kernel
// Minimal ET implementation for inplace F32 SET into a contiguous destination
// using a contiguous F32 source view and explicit destination view strides.
//
// Supported shape family:
// - dst/base is contiguous F32
// - src1 is contiguous F32
// - src1.ne[0] is cacheline-aligned (multiple of 16 floats)
// - destination view strides/offset are cacheline-aligned
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_ਸੈੱਟ_params {
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            ਵਿੱਥ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ copy_ਪੰਕਤੀ_aligned(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int32_t n) {
    ਲਈ (int32_t i = 0; i < n; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src[i])
            : "f11");
    }
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਸੈੱਟ_params * params, ਖਾਲੀ * env) {
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

    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਜੇ (!src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne10 = src1->ne[0];
    ਅਚਲ int64_t ne11 = src1->ne[1];
    ਅਚਲ int64_t ne12 = src1->ne[2];
    ਅਚਲ int64_t ne13 = src1->ne[3];

    ਜੇ (src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || ne10 % 16 != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t nb11 = src1->nb[1];
    ਅਚਲ int64_t nb12 = src1->nb[2];
    ਅਚਲ int64_t nb13 = src1->nb[3];

    ਅਚਲ int64_t dnb1   = params->nb1;
    ਅਚਲ int64_t dnb2   = params->nb2;
    ਅਚਲ int64_t dnb3   = params->nb3;
    ਅਚਲ int64_t ਵਿੱਥ = params->ਵਿੱਥ;

    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne11 * ne12 * ne13;

    ਲਈ (int64_t ਪੰਕਤੀ = thread_id; ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t i1 = ਪੰਕਤੀ % ne11;
        ਅਚਲ int64_t i2 = (ਪੰਕਤੀ / ne11) % ne12;
        ਅਚਲ int64_t i3 = ਪੰਕਤੀ / (ne11 * ne12);

        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + i1 * nb11 + i2 * nb12 + i3 * nb13);
        ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + ਵਿੱਥ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, (int32_t) ne10);
    }

    ਵਾਪਸ 0;
}
