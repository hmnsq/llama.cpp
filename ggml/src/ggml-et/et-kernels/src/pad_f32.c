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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_pad_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_copy_f32(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int32_t n) {
    int32_t       i       = 0;
    ಸ್ಥಿರಾಂಕ int32_t vec_ಕೊನೆ = (n / 8) * 8;
    ಗಾಗಿ (; i < vec_ಕೊನೆ; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst[i])
            : [s] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src[i])
            : "f10");
    }
    ಗಾಗಿ (; i < n; i++) {
        dst[i] = src[i];
    }
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_pad_params * params, ಶೂನ್ಯ * env) {
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

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Dst dimensions
    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ಸ್ಥಿರಾಂಕ int64_t nb1_src = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2_src = src0->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb3_src = src0->nb[3];

    // Padding values
    ಸ್ಥಿರಾಂಕ int32_t lp1 = params->lp[1];
    ಸ್ಥಿರಾಂಕ int32_t rp1 = params->rp[1];
    ಸ್ಥಿರಾಂಕ int32_t lp2 = params->lp[2];
    ಸ್ಥಿರಾಂಕ int32_t rp2 = params->rp[2];
    ಸ್ಥಿರಾಂಕ int32_t lp3 = params->lp[3];
    ಸ್ಥಿರಾಂಕ int32_t rp3 = params->rp[3];

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು     = ne1 * ne2 * ne3;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = ne0 * ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (ಒಟ್ಟು_elements == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    ದಶಮಾಂಶ zero = 0.0f;
    __asm__ ಅಸ್ಥಿರ("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    ಆದರೆ (ne0 % 16 == 0) {
        ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = thread_id; ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು += ಅಂಕೆ_ಎಳೆಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಡ್ಡಸಾಲು / (ne1 * ne2);
            ಸ್ಥಿರಾಂಕ int64_t i2 = (ಅಡ್ಡಸಾಲು / ne1) % ne2;
            ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಡ್ಡಸಾಲು % ne1;

            ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು * ne0;

            ಆದರೆ (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, (int32_t) ne0);
            } ಇಲ್ಲದಿದ್ದರೆ {
                int64_t       i       = 0;
                ಸ್ಥಿರಾಂಕ int64_t vec_ಕೊನೆ = (ne0 / 8) * 8;
                ಗಾಗಿ (; i < vec_ಕೊನೆ; i += 8) {
                    __asm__ ಅಸ್ಥಿರ("fsw.ps f12, %[d]\n" : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[i])::"f12");
                }
            }
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl = 16;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl     = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

    ಸ್ಥಿರಾಂಕ int64_t ne1_ದತ್ತಾಂಶ_ಕೊನೆ = ne1 - rp1;
    ಸ್ಥಿರಾಂಕ int64_t ne2_ದತ್ತಾಂಶ_ಕೊನೆ = ne2 - rp2;
    ಸ್ಥಿರಾಂಕ int64_t ne3_ದತ್ತಾಂಶ_ಕೊನೆ = ne3 - rp3;

    ಗಾಗಿ (int64_t cl = thread_id; cl < ಒಟ್ಟು_cl; cl += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t elem_ಆರಂಭ = cl * elems_ಪ್ರತಿ_cl;
        int64_t       elem_ಕೊನೆ   = elem_ಆರಂಭ + elems_ಪ್ರತಿ_cl;
        ಆದರೆ (elem_ಕೊನೆ > ಒಟ್ಟು_elements) {
            elem_ಕೊನೆ = ಒಟ್ಟು_elements;
        }

        ಗಾಗಿ (int64_t idx = elem_ಆರಂಭ; idx < elem_ಕೊನೆ; idx++) {
            ಸ್ಥಿರಾಂಕ int64_t i0   = idx % ne0;
            ಸ್ಥಿರಾಂಕ int64_t rem  = idx / ne0;
            ಸ್ಥಿರಾಂಕ int64_t i1   = rem % ne1;
            ಸ್ಥಿರಾಂಕ int64_t rem2 = rem / ne1;
            ಸ್ಥಿರಾಂಕ int64_t i2   = rem2 % ne2;
            ಸ್ಥಿರಾಂಕ int64_t i3   = rem2 / ne2;

            ಆದರೆ (i1 >= lp1 && i1 < ne1_ದತ್ತಾಂಶ_ಕೊನೆ && i2 >= lp2 && i2 < ne2_ದತ್ತಾಂಶ_ಕೊನೆ && i3 >= lp3 && i3 < ne3_ದತ್ತಾಂಶ_ಕೊನೆ) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sp = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_ದತ್ತಾಂಶ[idx]    = *sp;
            } ಇಲ್ಲದಿದ್ದರೆ {
                dst_ದತ್ತಾಂಶ[idx] = 0.0f;
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
