//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_cumsum_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_cumsum_params * params, ಶೂನ್ಯ * env) {
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

    ಸ್ಥಿರಾಂಕ int64_t ne0 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ size_t snb0 = src0->nb[0];
    ಸ್ಥಿರಾಂಕ size_t snb1 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t snb2 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ size_t snb3 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ size_t dnb0 = dst->nb[0];
    ಸ್ಥಿರಾಂಕ size_t dnb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t dnb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ size_t dnb3 = dst->nb[3];

    ಆದರೆ (snb0 != ಗಾತ್ರ(ದಶಮಾಂಶ) || dnb0 != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು     = ne1 * ne2 * ne3;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು = et_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_cacheline_ಗುಂಪು(ne0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_groups   = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;

    ಗಾಗಿ (int64_t grp = thread_id; grp < ಒಟ್ಟು_groups; grp += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = grp * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
        int64_t       ಅಡ್ಡಸಾಲು_ಕೊನೆ   = ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
        ಆದರೆ (ಅಡ್ಡಸಾಲು_ಕೊನೆ > ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            ಅಡ್ಡಸಾಲು_ಕೊನೆ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        }

        ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ++ಅಡ್ಡಸಾಲು) {
            int64_t i1 = ಅಡ್ಡಸಾಲು % ne1;
            int64_t i2 = (ಅಡ್ಡಸಾಲು / ne1) % ne2;
            int64_t i3 = ಅಡ್ಡಸಾಲು / (ne1 * ne2);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * snb1 + i2 * snb2 + i3 * snb3);
            ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            ದಶಮಾಂಶ acc = 0.0f;
            ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_ಅಡ್ಡಸಾಲು[i0];
                dst_ಅಡ್ಡಸಾಲು[i0] = acc;
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
