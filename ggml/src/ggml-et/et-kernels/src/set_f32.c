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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_ಗಣ_params {
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            ಸ್ಥಾನಾಂತರ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ copy_ಅಡ್ಡಸಾಲು_aligned(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int32_t n) {
    ಗಾಗಿ (int32_t i = 0; i < n; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src[i])
            : "f11");
    }
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ಗಣ_params * params, ಶೂನ್ಯ * env) {
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

    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ಆದರೆ (!src1_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne13 = src1->ne[3];

    ಆದರೆ (src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || ne10 % 16 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t nb11 = src1->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb12 = src1->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb13 = src1->nb[3];

    ಸ್ಥಿರಾಂಕ int64_t dnb1   = params->nb1;
    ಸ್ಥಿರಾಂಕ int64_t dnb2   = params->nb2;
    ಸ್ಥಿರಾಂಕ int64_t dnb3   = params->nb3;
    ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ = params->ಸ್ಥಾನಾಂತರ;

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne11 * ne12 * ne13;

    ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = thread_id; ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಡ್ಡಸಾಲು % ne11;
        ಸ್ಥಿರಾಂಕ int64_t i2 = (ಅಡ್ಡಸಾಲು / ne11) % ne12;
        ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಡ್ಡಸಾಲು / (ne11 * ne12);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + i1 * nb11 + i2 * nb12 + i3 * nb13);
        ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, (int32_t) ne10);
    }

    ಹಿಂತಿರುಗಿ 0;
}
