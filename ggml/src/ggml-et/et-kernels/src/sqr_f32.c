//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
ರಚನೆ ggml_et_sqr_params {
    ರಚನೆ ggml_tensor src0;  // F32 input tensor
    ರಚನೆ ggml_tensor dst;   // F32 output tensor
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_sqr_params * params, ಶೂನ್ಯ * env) {
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
        ಹಿಂತಿರುಗಿ -1;  // Invalid pointer
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported type combination
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t elements_ಪ್ರತಿ_cacheline = 16;  // 64 bytes / 4 bytes per float
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cachelines       = (ಒಟ್ಟು_elements + elements_ಪ್ರತಿ_cacheline - 1) / elements_ಪ್ರತಿ_cacheline;

    ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cachelines + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
    int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
    ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cachelines) {
        cl_ಕೊನೆ = ಒಟ್ಟು_cachelines;
    }

    ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cachelines) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ int64_t elem_ಆರಂಭ = cl_ಆರಂಭ * elements_ಪ್ರತಿ_cacheline;
    int64_t       elem_ಕೊನೆ   = cl_ಕೊನೆ * elements_ಪ್ರತಿ_cacheline;
    ಆದರೆ (elem_ಕೊನೆ > ಒಟ್ಟು_elements) {
        elem_ಕೊನೆ = ಒಟ್ಟು_elements;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = src0_ದತ್ತಾಂಶ + elem_ಆರಂಭ;
    ದಶಮಾಂಶ *       dst_ptr = dst_ದತ್ತಾಂಶ + elem_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ int32_t ಎಣಿಕೆ_2   = (int32_t) (elem_ಕೊನೆ - elem_ಆರಂಭ);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    ಗಾಗಿ (int32_t i0 = 0; i0 < ಎಣಿಕೆ_2; i0 += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [ಫಲಿತಾಂಶ] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    ಹಿಂತಿರುಗಿ 0;
}
