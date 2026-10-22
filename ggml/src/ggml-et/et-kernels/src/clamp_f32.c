//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_clamp_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input  (contiguous)
    ਬਣਤਰ ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    ਦਸ਼ਮਲਵ              min_val;
    ਦਸ਼ਮਲਵ              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ clamp_ਖੰਡ_f32(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ min_val, ਦਸ਼ਮਲਵ max_val, int32_t n) {
    int32_t       i       = 0;
    ਅਚਲ int32_t vec_ਸਿਰਾ = (n / 8) * 8;

    ਜੇ (vec_ਸਿਰਾ > 0) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
        __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
        __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

        ਲਈ (; i < vec_ਸਿਰਾ; i += 8) {
            __asm__ ਅਸਥਿਰ(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])
                : [s] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    }

    ਲਈ (; i < n; i++) {
        ਦਸ਼ਮਲਵ v = src[i];
        ਜੇ (v < min_val) {
            v = min_val;
        }
        ਜੇ (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_clamp_params * params, ਖਾਲੀ * env) {
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

    ਅਚਲ int64_t ਕੁੱਲ_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਜੇ (ਕੁੱਲ_elements <= 0) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ min_val = params->min_val;
    ਅਚਲ ਦਸ਼ਮਲਵ max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl = 16;
    ਅਚਲ int64_t ਕੁੱਲ_cl     = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

    ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cl + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = (int64_t) thread_id * cl_ਪ੍ਰਤੀ_thread;
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

    clamp_ਖੰਡ_f32(dst_ਅੰਕੜੇ + es, src0_ਅੰਕੜੇ + es, min_val, max_val, (int32_t) (ee - es));
    ਵਾਪਸ 0;
}
