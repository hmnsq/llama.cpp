//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_fill_params {
    ರಚನೆ ggml_tensor dst;  // F32 output tensor (contiguous)
    ದಶಮಾಂಶ              c;    // Constant value to fill
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_fill_params * params, ಶೂನ್ಯ * env) {
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

    ರಚನೆ ggml_tensor * dst = &params->dst;

    ಆದರೆ (dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ಆದರೆ (!dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    ಆದರೆ (ಒಟ್ಟು_elements == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl  = 16;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl      = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;
    ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cl + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
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

    // Broadcast constant to all SIMD lanes
    ದಶಮಾಂಶ c = params->c;
    __asm__ ಅಸ್ಥಿರ("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ಸ್ಥಿರಾಂಕ int64_t vec_ಕೊನೆ = es + ((ee - es) / 8) * 8;
    ಗಾಗಿ (; i < vec_ಕೊನೆ; i += 8) {
        __asm__ ಅಸ್ಥಿರ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ದತ್ತಾಂಶ[i])::"f10");
    }
    // Scalar tail
    ಗಾಗಿ (; i < ee; i++) {
        dst_ದತ್ತಾಂಶ[i] = c;
    }

    ಹಿಂತಿರುಗಿ 0;
}
