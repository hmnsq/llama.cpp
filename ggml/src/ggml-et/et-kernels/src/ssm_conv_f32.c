#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_ssm_conv_params {
    संरचना ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    संरचना ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    संरचना ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

पूर्णाङ्क entry_point(संरचना ggml_et_ssm_conv_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * src0_data = (ध्रुव भिन्न *) src0->data;
    ध्रुव भिन्न * src1_data = (ध्रुव भिन्न *) src1->data;
    भिन्न *       dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !src1_data || !dst_data) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t nc  = src1->ne[0];
    ध्रुव int64_t ncs = src0->ne[0];
    ध्रुव int64_t nr  = src0->ne[1];
    ध्रुव int64_t n_t = dst->ne[1];
    ध्रुव int64_t n_s = dst->ne[2];

    यदि (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != परिमाण(भिन्न) ||
        src1->nb[0] != परिमाण(भिन्न) || dst->nb[0] != परिमाण(भिन्न) || src0->nb[1] != (size_t) ncs * परिमाण(भिन्न) ||
        src1->nb[1] != (size_t) nc * परिमाण(भिन्न)) {
        प्रत्यागम -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ध्रुव int64_t chunk    = 16;
    ध्रुव int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ चञ्चल("mova.x.m %0" : "=r"(saved_mask));
    __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

    प्रति (int64_t i3 = 0; i3 < n_s; ++i3) {
        प्रति (int64_t i2 = 0; i2 < n_t; ++i2) {
            ध्रुव भिन्न * s = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            भिन्न *       x = (भिन्न *) ((वर्ण *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            प्रति (int64_t ci = thread_id; ci < n_chunks; ci += अंक_threads) {
                ध्रुव int64_t i1_start = ci * chunk;
                ध्रुव int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                प्रति (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    भिन्न tmp_s[8], tmp_c[8];
                    भिन्न acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    प्रति (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        प्रति (पूर्णाङ्क j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ चञ्चल(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(भिन्न (*)[8]) acc)
                            : [acc] "m"(*(ध्रुव भिन्न (*)[8]) acc), [sv] "m"(*(ध्रुव भिन्न (*)[8]) tmp_s),
                              [cv] "m"(*(ध्रुव भिन्न (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ चञ्चल(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(भिन्न (*)[8])(x + i1))
                        : [acc] "m"(*(ध्रुव भिन्न (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                प्रति (; i1 < i1_end; ++i1) {
                    ध्रुव भिन्न * c     = src1_data + i1 * nc;
                    ध्रुव भिन्न * s_row = s + i1 * ncs;
                    भिन्न         sumf  = 0.0f;
                    प्रति (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_mask));

    प्रत्यागम 0;
}
