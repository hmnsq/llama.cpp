#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_ssm_conv_params {
    ಸಂರಚನಾ ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ಸಂರಚನಾ ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ಸಂರಚನಾ ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_ssm_conv_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಭಿನ್ನ * src0_data = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಧ್ರುವ ಭಿನ್ನ * src1_data = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ *       dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !src1_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t nc  = src1->ne[0];
    ಧ್ರುವ int64_t ncs = src0->ne[0];
    ಧ್ರುವ int64_t nr  = src0->ne[1];
    ಧ್ರುವ int64_t n_t = dst->ne[1];
    ಧ್ರುವ int64_t n_s = dst->ne[2];

    ಯದಿ (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) ||
        src1->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || dst->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || src0->nb[1] != (size_t) ncs * ಪರಿಮಾಣ(ಭಿನ್ನ) ||
        src1->nb[1] != (size_t) nc * ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ಧ್ರುವ int64_t chunk    = 16;
    ಧ್ರುವ int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(saved_mask));
    __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");

    ಪ್ರತಿ (int64_t i3 = 0; i3 < n_s; ++i3) {
        ಪ್ರತಿ (int64_t i2 = 0; i2 < n_t; ++i2) {
            ಧ್ರುವ ಭಿನ್ನ * s = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            ಭಿನ್ನ *       x = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            ಪ್ರತಿ (int64_t ci = thread_id; ci < n_chunks; ci += ಅಂಕೆ_threads) {
                ಧ್ರುವ int64_t i1_start = ci * chunk;
                ಧ್ರುವ int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                ಪ್ರತಿ (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    ಭಿನ್ನ tmp_s[8], tmp_c[8];
                    ಭಿನ್ನ acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    ಪ್ರತಿ (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ ಚಞ್ಚಲ(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(ಭಿನ್ನ (*)[8]) acc)
                            : [acc] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) acc), [sv] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) tmp_s),
                              [cv] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ ಚಞ್ಚಲ(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(ಭಿನ್ನ (*)[8])(x + i1))
                        : [acc] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                ಪ್ರತಿ (; i1 < i1_end; ++i1) {
                    ಧ್ರುವ ಭಿನ್ನ * c     = src1_data + i1 * nc;
                    ಧ್ರುವ ಭಿನ್ನ * s_row = s + i1 * ncs;
                    ಭಿನ್ನ         sumf  = 0.0f;
                    ಪ್ರತಿ (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));

    ಪ್ರತ್ಯಾಗಮ 0;
}
