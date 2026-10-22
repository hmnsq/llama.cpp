#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_ssm_conv_params {
    ସଂରଚନା ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ସଂରଚନା ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ସଂରଚନା ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_ssm_conv_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * src1 = &params->src1;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ଦଶମିକ * src0_data = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଅଚଳ ଦଶମିକ * src1_data = (ଅଚଳ ଦଶମିକ *) src1->data;
    ଦଶମିକ *       dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !src1_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t nc  = src1->ne[0];
    ଅଚଳ int64_t ncs = src0->ne[0];
    ଅଚଳ int64_t nr  = src0->ne[1];
    ଅଚଳ int64_t n_t = dst->ne[1];
    ଅଚଳ int64_t n_s = dst->ne[2];

    ଯଦି (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != ଆକାର(ଦଶମିକ) ||
        src1->nb[0] != ଆକାର(ଦଶମିକ) || dst->nb[0] != ଆକାର(ଦଶମିକ) || src0->nb[1] != (size_t) ncs * ଆକାର(ଦଶମିକ) ||
        src1->nb[1] != (size_t) nc * ଆକାର(ଦଶମିକ)) {
        ଫେରନ୍ତୁ -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ଅଚଳ int64_t chunk    = 16;
    ଅଚଳ int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(saved_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ପାଇଁ (int64_t i3 = 0; i3 < n_s; ++i3) {
        ପାଇଁ (int64_t i2 = 0; i2 < n_t; ++i2) {
            ଅଚଳ ଦଶମିକ * s = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            ଦଶମିକ *       x = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            ପାଇଁ (int64_t ci = thread_id; ci < n_chunks; ci += ଅଙ୍କ_threads) {
                ଅଚଳ int64_t i1_start = ci * chunk;
                ଅଚଳ int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                ପାଇଁ (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    ଦଶମିକ tmp_s[8], tmp_c[8];
                    ଦଶମିକ acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    ପାଇଁ (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ ଅସ୍ଥିର(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(ଦଶମିକ (*)[8]) acc)
                            : [acc] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) acc), [sv] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) tmp_s),
                              [cv] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ ଅସ୍ଥିର(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(ଦଶମିକ (*)[8])(x + i1))
                        : [acc] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                ପାଇଁ (; i1 < i1_end; ++i1) {
                    ଅଚଳ ଦଶମିକ * c     = src1_data + i1 * nc;
                    ଅଚଳ ଦଶମିକ * s_row = s + i1 * ncs;
                    ଦଶମିକ         sumf  = 0.0f;
                    ପାଇଁ (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(saved_mask));

    ଫେରନ୍ତୁ 0;
}
