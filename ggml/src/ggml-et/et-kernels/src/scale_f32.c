//******************************************************************************
// Scale F32 Kernel
// dst[i] = src0[i] * scale + bias
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_scale_params {
    ರಚನೆ ggml_tensor src0;   // F32 input tensor
    ರಚನೆ ggml_tensor dst;    // F32 output tensor
    ದಶಮಾಂಶ              scale;  // Scale factor
    ದಶಮಾಂಶ              bias;   // Bias (additive offset)
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_scale_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

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


    ದಶಮಾಂಶ scale = params->scale;
    ದಶಮಾಂಶ bias  = params->bias;

    // Total elements across all dimensions
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Cache line = 64 bytes = 16 floats, but vector width = 8 floats
    // Parallelize at cache line granularity (16 floats)
    ಸ್ಥಿರಾಂಕ int64_t elements_ಪ್ರತಿ_cacheline = 16;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cachelines       = (ಒಟ್ಟು_elements + elements_ಪ್ರತಿ_cacheline - 1) / elements_ಪ್ರತಿ_cacheline;

    int64_t cachelines_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cachelines + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    int64_t ಆರಂಭ_cacheline       = thread_id * cachelines_ಪ್ರತಿ_thread;
    int64_t ಕೊನೆ_cacheline         = ಆರಂಭ_cacheline + cachelines_ಪ್ರತಿ_thread;

    ಆದರೆ (ಕೊನೆ_cacheline > ಒಟ್ಟು_cachelines) {
        ಕೊನೆ_cacheline = ಒಟ್ಟು_cachelines;
    }

    ಆದರೆ (ಆರಂಭ_cacheline >= ಒಟ್ಟು_cachelines) {
        ಹಿಂತಿರುಗಿ 0;
    }

    int64_t ಆರಂಭ_elem = ಆರಂಭ_cacheline * elements_ಪ್ರತಿ_cacheline;
    int64_t ಕೊನೆ_elem   = ಕೊನೆ_cacheline * elements_ಪ್ರತಿ_cacheline;
    ಆದರೆ (ಕೊನೆ_elem > ಒಟ್ಟು_elements) {
        ಕೊನೆ_elem = ಒಟ್ಟು_elements;
    }

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");
    __asm__ ಅಸ್ಥಿರ("fbc.ps f20, %[scale_ptr]\n" : : [scale_ptr] "m"(scale) : "f20");
    __asm__ ಅಸ್ಥಿರ("fbc.ps f21, %[bias_ptr]\n" : : [bias_ptr] "m"(bias) : "f21");

    ಗಾಗಿ (int64_t i = ಆರಂಭ_elem; i < ಕೊನೆ_elem; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps f10, %[src]\n"
            "fmadd.ps f10, f10, f20, f21\n"  // dst = src*scale + bias
            "fsw.ps f10, %[dst_out]\n"
            : [dst_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ದತ್ತಾಂಶ[i])
            : [src] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src0_ದತ್ತಾಂಶ[i])
            : "f10", "f20", "f21");
    }
    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));

    ಹಿಂತಿರುಗಿ 0;
}
