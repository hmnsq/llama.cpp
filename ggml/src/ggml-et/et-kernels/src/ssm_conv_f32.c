#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_ssm_conv_params {
    ഘടന ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ഘടന ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ഘടന ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_ssm_conv_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * src1 = &params->src1;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം ദശാംശം * src0_data = (സ്ഥിരം ദശാംശം *) src0->data;
    സ്ഥിരം ദശാംശം * src1_data = (സ്ഥിരം ദശാംശം *) src1->data;
    ദശാംശം *       dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !src1_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t nc  = src1->ne[0];
    സ്ഥിരം int64_t ncs = src0->ne[0];
    സ്ഥിരം int64_t nr  = src0->ne[1];
    സ്ഥിരം int64_t n_t = dst->ne[1];
    സ്ഥിരം int64_t n_s = dst->ne[2];

    എങ്കിൽ (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != വലുപ്പം(ദശാംശം) ||
        src1->nb[0] != വലുപ്പം(ദശാംശം) || dst->nb[0] != വലുപ്പം(ദശാംശം) || src0->nb[1] != (size_t) ncs * വലുപ്പം(ദശാംശം) ||
        src1->nb[1] != (size_t) nc * വലുപ്പം(ദശാംശം)) {
        തിരികെ -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    സ്ഥിരം int64_t chunk    = 16;
    സ്ഥിരം int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(saved_mask));
    __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");

    വേണ്ടി (int64_t i3 = 0; i3 < n_s; ++i3) {
        വേണ്ടി (int64_t i2 = 0; i2 < n_t; ++i2) {
            സ്ഥിരം ദശാംശം * s = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            ദശാംശം *       x = (ദശാംശം *) ((അക്ഷരം *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            വേണ്ടി (int64_t ci = thread_id; ci < n_chunks; ci += അക്കം_threads) {
                സ്ഥിരം int64_t i1_start = ci * chunk;
                സ്ഥിരം int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                വേണ്ടി (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    ദശാംശം tmp_s[8], tmp_c[8];
                    ദശാംശം acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    വേണ്ടി (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ അസ്ഥിര(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(ദശാംശം (*)[8]) acc)
                            : [acc] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) acc), [sv] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) tmp_s),
                              [cv] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ അസ്ഥിര(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(ദശാംശം (*)[8])(x + i1))
                        : [acc] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                വേണ്ടി (; i1 < i1_end; ++i1) {
                    സ്ഥിരം ദശാംശം * c     = src1_data + i1 * nc;
                    സ്ഥിരം ദശാംശം * s_row = s + i1 * ncs;
                    ദശാംശം         sumf  = 0.0f;
                    വേണ്ടി (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(saved_mask));

    തിരികെ 0;
}
