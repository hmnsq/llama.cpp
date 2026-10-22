//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_diag_params {
    ರಚನೆ ggml_tensor src0;  // F32 input vector
    ರಚನೆ ggml_tensor dst;   // F32 output diagonal matrix
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_diag_params * params, ಶೂನ್ಯ * env) {
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

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];  // N (number of rows)
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    ದಶಮಾಂಶ zero = 0.0f;
    __asm__ ಅಸ್ಥಿರ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = thread_id; ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು += ಅಂಕೆ_ಎಳೆಗಳು) {
        int64_t i1 = ಅಡ್ಡಸಾಲು % ne1;
        int64_t i2 = (ಅಡ್ಡಸಾಲು / ne1) % ne2;
        int64_t i3 = ಅಡ್ಡಸಾಲು / (ne1 * ne2);

        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ಸ್ಥಿರಾಂಕ int64_t vec_ಕೊನೆ = (ne0 / 8) * 8;
        ಗಾಗಿ (; i0 < vec_ಕೊನೆ; i0 += 8) {
            __asm__ ಅಸ್ಥಿರ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[i0])::"f10");
        }
        ಗಾಗಿ (; i0 < ne0; i0++) {
            dst_ಅಡ್ಡಸಾಲು[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i2 * nb02 + i3 * nb03);
        dst_ಅಡ್ಡಸಾಲು[i1]           = src_ptr[i1];
    }

    ಹಿಂತಿರುಗಿ 0;
}
