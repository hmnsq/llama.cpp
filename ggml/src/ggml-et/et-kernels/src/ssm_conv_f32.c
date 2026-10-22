#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_ssm_conv_params {
    काष्ठा ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    काष्ठा ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    काष्ठा ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

पूर्णांक entry_point(काष्ठा ggml_et_ssm_conv_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न * src0_तथ्य = (स्थिर भग्न *) src0->तथ्य;
    स्थिर भग्न * src1_तथ्य = (स्थिर भग्न *) src1->तथ्य;
    भग्न *       dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !src1_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t nc  = src1->ne[0];
    स्थिर int64_t ncs = src0->ne[0];
    स्थिर int64_t nr  = src0->ne[1];
    स्थिर int64_t n_t = dst->ne[1];
    स्थिर int64_t n_s = dst->ne[2];

    अगर (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != माप(भग्न) ||
        src1->nb[0] != माप(भग्न) || dst->nb[0] != माप(भग्न) || src0->nb[1] != (माप_प्रकार) ncs * माप(भग्न) ||
        src1->nb[1] != (माप_प्रकार) nc * माप(भग्न)) {
        वापस -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    स्थिर int64_t chunk    = 16;
    स्थिर int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    अचिन्हित दीर्घ saved_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

    क्रम (int64_t i3 = 0; i3 < n_s; ++i3) {
        क्रम (int64_t i2 = 0; i2 < n_t; ++i2) {
            स्थिर भग्न * s = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i2 * src0->nb[0] + i3 * src0->nb[2]);
            भग्न *       x = (भग्न *) ((अक्षर *) dst_तथ्य + i2 * dst->nb[1] + i3 * dst->nb[2]);

            क्रम (int64_t ci = thread_id; ci < n_chunks; ci += अंक_threads) {
                स्थिर int64_t i1_start = ci * chunk;
                स्थिर int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                क्रम (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    भग्न tmp_s[8], tmp_c[8];
                    भग्न acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    क्रम (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        क्रम (पूर्णांक j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_तथ्य[(i1 + j) * nc + i0];
                        }

                        __asm__ अस्थिर(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(भग्न (*)[8]) acc)
                            : [acc] "m"(*(स्थिर भग्न (*)[8]) acc), [sv] "m"(*(स्थिर भग्न (*)[8]) tmp_s),
                              [cv] "m"(*(स्थिर भग्न (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ अस्थिर(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(भग्न (*)[8])(x + i1))
                        : [acc] "m"(*(स्थिर भग्न (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                क्रम (; i1 < i1_end; ++i1) {
                    स्थिर भग्न * c     = src1_तथ्य + i1 * nc;
                    स्थिर भग्न * s_row = s + i1 * ncs;
                    भग्न         sumf  = 0.0f;
                    क्रम (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));

    वापस 0;
}
