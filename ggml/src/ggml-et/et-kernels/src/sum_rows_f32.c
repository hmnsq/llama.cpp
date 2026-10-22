//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_params {
    ರಚನೆ ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ರಚನೆ ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_params * params, ಶೂನ್ಯ * env) {
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


    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ size_t nb03 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ size_t nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne01 * ne02 * ne03;

    ಗಾಗಿ (int64_t ir = thread_id; ir < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ir += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne02 * ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ದಶಮಾಂಶ *       dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        ದಶಮಾಂಶ zero = 0.0f;
        __asm__ ಅಸ್ಥಿರ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ಗಾಗಿ (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ಅಡ್ಡಸಾಲು[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        ದಶಮಾಂಶ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(ಅಡ್ಡಸಾಲು_ಮೊತ್ತ)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, ಅಡ್ಡಸಾಲು_ಮೊತ್ತ);
    }

    ಹಿಂತಿರುಗಿ 0;
}
