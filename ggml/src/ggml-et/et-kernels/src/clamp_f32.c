//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_clamp_params {
    ರಚನೆ ggml_tensor src0;  // F32 input  (contiguous)
    ರಚನೆ ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    ದಶಮಾಂಶ              min_val;
    ದಶಮಾಂಶ              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ clamp_ಖಂಡ_f32(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ min_val, ದಶಮಾಂಶ max_val, int32_t n) {
    int32_t       i       = 0;
    ಸ್ಥಿರಾಂಕ int32_t vec_ಕೊನೆ = (n / 8) * 8;

    ಆದರೆ (vec_ಕೊನೆ > 0) {
        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
        __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
        __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

        ಗಾಗಿ (; i < vec_ಕೊನೆ; i += 8) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst[i])
                : [s] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
    }

    ಗಾಗಿ (; i < n; i++) {
        ದಶಮಾಂಶ v = src[i];
        ಆದರೆ (v < min_val) {
            v = min_val;
        }
        ಆದರೆ (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_clamp_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಆದರೆ (ಒಟ್ಟು_elements <= 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min_val = params->min_val;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl = 16;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl     = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

    ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cl + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = (int64_t) thread_id * cl_ಪ್ರತಿ_thread;
    int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
    ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cl) {
        cl_ಕೊನೆ = ಒಟ್ಟು_cl;
    }
    ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cl) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ int64_t es = cl_ಆರಂಭ * elems_ಪ್ರತಿ_cl;
    int64_t       ee = cl_ಕೊನೆ * elems_ಪ್ರತಿ_cl;
    ಆದರೆ (ee > ಒಟ್ಟು_elements) {
        ee = ಒಟ್ಟು_elements;
    }

    clamp_ಖಂಡ_f32(dst_ದತ್ತಾಂಶ + es, src0_ದತ್ತಾಂಶ + es, min_val, max_val, (int32_t) (ee - es));
    ಹಿಂತಿರುಗಿ 0;
}
