#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_ssm_conv_params {
    संरचना ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    संरचना ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    संरचना ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

पूर्णांक entry_point(संरचना ggml_et_ssm_conv_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक दशांश * src0_data = (स्थिरांक दशांश *) src0->data;
    स्थिरांक दशांश * src1_data = (स्थिरांक दशांश *) src1->data;
    दशांश *       dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !src1_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int64_t nc  = src1->ne[0];
    स्थिरांक int64_t ncs = src0->ne[0];
    स्थिरांक int64_t nr  = src0->ne[1];
    स्थिरांक int64_t n_t = dst->ne[1];
    स्थिरांक int64_t n_s = dst->ne[2];

    जर (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != आकार(दशांश) ||
        src1->nb[0] != आकार(दशांश) || dst->nb[0] != आकार(दशांश) || src0->nb[1] != (माप_प्रकार) ncs * आकार(दशांश) ||
        src1->nb[1] != (माप_प्रकार) nc * आकार(दशांश)) {
        परत -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    स्थिरांक int64_t chunk    = 16;
    स्थिरांक int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    अचिन्हित दीर्घ saved_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

    साठी (int64_t i3 = 0; i3 < n_s; ++i3) {
        साठी (int64_t i2 = 0; i2 < n_t; ++i2) {
            स्थिरांक दशांश * s = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            दशांश *       x = (दशांश *) ((अक्षर *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            साठी (int64_t ci = thread_id; ci < n_chunks; ci += अंक_threads) {
                स्थिरांक int64_t i1_start = ci * chunk;
                स्थिरांक int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                साठी (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    दशांश tmp_s[8], tmp_c[8];
                    दशांश acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    साठी (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        साठी (पूर्णांक j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ अस्थिर(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(दशांश (*)[8]) acc)
                            : [acc] "m"(*(स्थिरांक दशांश (*)[8]) acc), [sv] "m"(*(स्थिरांक दशांश (*)[8]) tmp_s),
                              [cv] "m"(*(स्थिरांक दशांश (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ अस्थिर(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(दशांश (*)[8])(x + i1))
                        : [acc] "m"(*(स्थिरांक दशांश (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                साठी (; i1 < i1_end; ++i1) {
                    स्थिरांक दशांश * c     = src1_data + i1 * nc;
                    स्थिरांक दशांश * s_row = s + i1 * ncs;
                    दशांश         sumf  = 0.0f;
                    साठी (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));

    परत 0;
}
