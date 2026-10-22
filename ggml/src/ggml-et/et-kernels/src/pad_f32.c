//******************************************************************************
// Bare Metal PAD F32 Kernel
// Zero-pads an F32 tensor along dimensions 1-3.
//
// Constraints:
//   - No dim0 padding (lp[0]==0, rp[0]==0)
//   - dst contiguous
//   - src nb[0] == 4 (dim0 contiguous for vectorized reads)
//   - Zero-pad only (no circular mode)
//
// Two paths:
//   Aligned (ne0 % 16 == 0): rows distributed across harts, vectorized.
//   Small   (16 % ne0 == 0): cache-line distributed, scalar per-element.
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_pad_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_copy_f32(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int32_t n) {
    int32_t       i       = 0;
    ਅਚਲ int32_t vec_ਸਿਰਾ = (n / 8) * 8;
    ਲਈ (; i < vec_ਸਿਰਾ; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])
            : [s] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src[i])
            : "f10");
    }
    ਲਈ (; i < n; i++) {
        dst[i] = src[i];
    }
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_pad_params * params, ਖਾਲੀ * env) {
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

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    // Dst dimensions
    ਅਚਲ int64_t ne0 = dst->ne[0];
    ਅਚਲ int64_t ne1 = dst->ne[1];
    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ਅਚਲ int64_t nb1_src = src0->nb[1];
    ਅਚਲ int64_t nb2_src = src0->nb[2];
    ਅਚਲ int64_t nb3_src = src0->nb[3];

    // Padding values
    ਅਚਲ int32_t lp1 = params->lp[1];
    ਅਚਲ int32_t rp1 = params->rp[1];
    ਅਚਲ int32_t lp2 = params->lp[2];
    ਅਚਲ int32_t rp2 = params->rp[2];
    ਅਚਲ int32_t lp3 = params->lp[3];
    ਅਚਲ int32_t rp3 = params->rp[3];

    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ     = ne1 * ne2 * ne3;
    ਅਚਲ int64_t ਕੁੱਲ_elements = ne0 * ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (ਕੁੱਲ_elements == 0) {
        ਵਾਪਸ 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    ਦਸ਼ਮਲਵ zero = 0.0f;
    __asm__ ਅਸਥਿਰ("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    ਜੇ (ne0 % 16 == 0) {
        ਲਈ (int64_t ਪੰਕਤੀ = thread_id; ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ += ਅੰਕ_ਧਾਗੇ) {
            ਅਚਲ int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);
            ਅਚਲ int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
            ਅਚਲ int64_t i1 = ਪੰਕਤੀ % ne1;

            ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + ਪੰਕਤੀ * ne0;

            ਜੇ (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, (int32_t) ne0);
            } ਨਹੀਂਤਾਂ {
                int64_t       i       = 0;
                ਅਚਲ int64_t vec_ਸਿਰਾ = (ne0 / 8) * 8;
                ਲਈ (; i < vec_ਸਿਰਾ; i += 8) {
                    __asm__ ਅਸਥਿਰ("fsw.ps f12, %[d]\n" : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[i])::"f12");
                }
            }
        }
        ਵਾਪਸ 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl = 16;
    ਅਚਲ int64_t ਕੁੱਲ_cl     = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

    ਅਚਲ int64_t ne1_ਅੰਕੜੇ_ਸਿਰਾ = ne1 - rp1;
    ਅਚਲ int64_t ne2_ਅੰਕੜੇ_ਸਿਰਾ = ne2 - rp2;
    ਅਚਲ int64_t ne3_ਅੰਕੜੇ_ਸਿਰਾ = ne3 - rp3;

    ਲਈ (int64_t cl = thread_id; cl < ਕੁੱਲ_cl; cl += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t elem_ਸ਼ੁਰੂ = cl * elems_ਪ੍ਰਤੀ_cl;
        int64_t       elem_ਸਿਰਾ   = elem_ਸ਼ੁਰੂ + elems_ਪ੍ਰਤੀ_cl;
        ਜੇ (elem_ਸਿਰਾ > ਕੁੱਲ_elements) {
            elem_ਸਿਰਾ = ਕੁੱਲ_elements;
        }

        ਲਈ (int64_t idx = elem_ਸ਼ੁਰੂ; idx < elem_ਸਿਰਾ; idx++) {
            ਅਚਲ int64_t i0   = idx % ne0;
            ਅਚਲ int64_t rem  = idx / ne0;
            ਅਚਲ int64_t i1   = rem % ne1;
            ਅਚਲ int64_t rem2 = rem / ne1;
            ਅਚਲ int64_t i2   = rem2 % ne2;
            ਅਚਲ int64_t i3   = rem2 / ne2;

            ਜੇ (i1 >= lp1 && i1 < ne1_ਅੰਕੜੇ_ਸਿਰਾ && i2 >= lp2 && i2 < ne2_ਅੰਕੜੇ_ਸਿਰਾ && i3 >= lp3 && i3 < ne3_ਅੰਕੜੇ_ਸਿਰਾ) {
                ਅਚਲ ਦਸ਼ਮਲਵ * sp = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_ਅੰਕੜੇ[idx]    = *sp;
            } ਨਹੀਂਤਾਂ {
                dst_ਅੰਕੜੇ[idx] = 0.0f;
            }
        }
    }

    ਵਾਪਸ 0;
}
