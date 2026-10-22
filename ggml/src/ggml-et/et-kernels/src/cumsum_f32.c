//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_cumsum_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_cumsum_params * params, ਖਾਲੀ * env) {
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

    ਅਚਲ int64_t ne0 = src0->ne[0];
    ਅਚਲ int64_t ne1 = src0->ne[1];
    ਅਚਲ int64_t ne2 = src0->ne[2];
    ਅਚਲ int64_t ne3 = src0->ne[3];

    ਅਚਲ size_t snb0 = src0->nb[0];
    ਅਚਲ size_t snb1 = src0->nb[1];
    ਅਚਲ size_t snb2 = src0->nb[2];
    ਅਚਲ size_t snb3 = src0->nb[3];

    ਅਚਲ size_t dnb0 = dst->nb[0];
    ਅਚਲ size_t dnb1 = dst->nb[1];
    ਅਚਲ size_t dnb2 = dst->nb[2];
    ਅਚਲ size_t dnb3 = dst->nb[3];

    ਜੇ (snb0 != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dnb0 != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ     = ne1 * ne2 * ne3;
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ = et_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_cacheline_ਸਮੂਹ(ne0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ int64_t ਕੁੱਲ_groups   = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;

    ਲਈ (int64_t grp = thread_id; grp < ਕੁੱਲ_groups; grp += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = grp * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
        int64_t       ਪੰਕਤੀ_ਸਿਰਾ   = ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
        ਜੇ (ਪੰਕਤੀ_ਸਿਰਾ > ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            ਪੰਕਤੀ_ਸਿਰਾ = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        }

        ਲਈ (int64_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; ++ਪੰਕਤੀ) {
            int64_t i1 = ਪੰਕਤੀ % ne1;
            int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
            int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * snb1 + i2 * snb2 + i3 * snb3);
            ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            ਦਸ਼ਮਲਵ acc = 0.0f;
            ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_ਪੰਕਤੀ[i0];
                dst_ਪੰਕਤੀ[i0] = acc;
            }
        }
    }

    ਵਾਪਸ 0;
}
