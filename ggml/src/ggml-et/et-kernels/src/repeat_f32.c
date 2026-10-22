//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

ਬਣਤਰ ggml_et_repeat_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input tensor (tile)
    ਬਣਤਰ ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
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

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਪ੍ਰਸਾਰਣ_scalar_aligned(ਦਸ਼ਮਲਵ * dst, ਦਸ਼ਮਲਵ val, int32_t n) {
    __asm__ ਅਸਥਿਰ("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    ਲਈ (int32_t i = 0; i < n; i += 8) {
        __asm__ ਅਸਥਿਰ("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])::"f11");
    }
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_repeat_params * params, ਖਾਲੀ * env) {
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

    ਅਚਲ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ਅਚਲ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    ਅਚਲ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    ਅਚਲ size_t dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    ਅਚਲ int32_t nr0 = (int32_t) (ne0 / ne00);
    ਅਚਲ int32_t nr1 = (int32_t) (ne1 / ne01);
    ਅਚਲ int32_t nr2 = (int32_t) (ne2 / ne02);
    ਅਚਲ int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne1 * ne2 * ne3;

    ਲਈ (int64_t ਪੰਕਤੀ = thread_id; ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ += ਅੰਕ_ਧਾਗੇ) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = ਪੰਕਤੀ % ne1;
        int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
        int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + k1 * nb01 + k2 * nb02 + k3 * nb03);
        ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ਜੇ (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            ਪ੍ਰਸਾਰਣ_scalar_aligned(dst_ਪੰਕਤੀ, *src_ਪੰਕਤੀ, (int32_t) ne0);
        } ਨਹੀਂਤਾਂ ਜੇ (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, (int32_t) ne00);
        } ਨਹੀਂਤਾਂ {
            // Tile ne00-sized chunks across dim 0
            ਲਈ (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ + i0 * ne00, src_ਪੰਕਤੀ, (int32_t) ne00);
            }
        }
    }

    ਵਾਪਸ 0;
}
