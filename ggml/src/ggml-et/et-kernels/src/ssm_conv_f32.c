#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_ssm_conv_params {
    ರಚನೆ ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ರಚನೆ ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ರಚನೆ ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ssm_conv_params * params, ಶೂನ್ಯ * env) {
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
    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t nc  = src1->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ncs = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nr  = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_t = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_s = dst->ne[2];

    ಆದರೆ (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) ||
        src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src0->nb[1] != (size_t) ncs * ಗಾತ್ರ(ದಶಮಾಂಶ) ||
        src1->nb[1] != (size_t) nc * ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ಸ್ಥಿರಾಂಕ int64_t chunk    = 16;
    ಸ್ಥಿರಾಂಕ int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ saved_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(saved_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ಗಾಗಿ (int64_t i3 = 0; i3 < n_s; ++i3) {
        ಗಾಗಿ (int64_t i2 = 0; i2 < n_t; ++i2) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i2 * src0->nb[0] + i3 * src0->nb[2]);
            ದಶಮಾಂಶ *       x = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i2 * dst->nb[1] + i3 * dst->nb[2]);

            ಗಾಗಿ (int64_t ci = thread_id; ci < n_chunks; ci += ಅಂಕೆ_ಎಳೆಗಳು) {
                ಸ್ಥಿರಾಂಕ int64_t i1_ಆರಂಭ = ci * chunk;
                ಸ್ಥಿರಾಂಕ int64_t i1_ಕೊನೆ   = i1_ಆರಂಭ + chunk < nr ? i1_ಆರಂಭ + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_ಆರಂಭ;
                ಗಾಗಿ (; i1 + 8 <= i1_ಕೊನೆ; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    ದಶಮಾಂಶ tmp_s[8], tmp_c[8];
                    ದಶಮಾಂಶ acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    ಗಾಗಿ (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_ದತ್ತಾಂಶ[(i1 + j) * nc + i0];
                        }

                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) acc)
                            : [acc] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) acc), [sv] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) tmp_s),
                              [cv] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ ಅಸ್ಥಿರ(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(ದಶಮಾಂಶ (*)[8])(x + i1))
                        : [acc] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                ಗಾಗಿ (; i1 < i1_ಕೊನೆ; ++i1) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * c     = src1_ದತ್ತಾಂಶ + i1 * nc;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s_ಅಡ್ಡಸಾಲು = s + i1 * ncs;
                    ದಶಮಾಂಶ         sumf  = 0.0f;
                    ಗಾಗಿ (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_ಅಡ್ಡಸಾಲು[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(saved_ಮುಖವಾಡ));

    ಹಿಂತಿರುಗಿ 0;
}
