#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

সংরচনা ggml_et_ssm_conv_params {
    সংরচনা ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    সংরচনা ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    সংরচনা ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_ssm_conv_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * src1 = &params->src1;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ধ্রুব ভিন্ন * src0_data = (ধ্রুব ভিন্ন *) src0->data;
    ধ্রুব ভিন্ন * src1_data = (ধ্রুব ভিন্ন *) src1->data;
    ভিন্ন *       dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !src1_data || !dst_data) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t nc  = src1->ne[0];
    ধ্রুব int64_t ncs = src0->ne[0];
    ধ্রুব int64_t nr  = src0->ne[1];
    ধ্রুব int64_t n_t = dst->ne[1];
    ধ্রুব int64_t n_s = dst->ne[2];

    যদি (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != পরিমাণ(ভিন্ন) ||
        src1->nb[0] != পরিমাণ(ভিন্ন) || dst->nb[0] != পরিমাণ(ভিন্ন) || src0->nb[1] != (size_t) ncs * পরিমাণ(ভিন্ন) ||
        src1->nb[1] != (size_t) nc * পরিমাণ(ভিন্ন)) {
        প্রত্যাগম -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ধ্রুব int64_t chunk    = 16;
    ধ্রুব int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ চঞ্চল("mova.x.m %0" : "=r"(saved_mask));
    __asm__ চঞ্চল("mov.m.x m0, x0, 0xFF");

    প্রতি (int64_t i3 = 0; i3 < n_s; ++i3) {
        প্রতি (int64_t i2 = 0; i2 < n_t; ++i2) {
            ধ্রুব ভিন্ন * s = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            ভিন্ন *       x = (ভিন্ন *) ((বর্ণ *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            প্রতি (int64_t ci = thread_id; ci < n_chunks; ci += অঙ্ক_threads) {
                ধ্রুব int64_t i1_start = ci * chunk;
                ধ্রুব int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                প্রতি (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    ভিন্ন tmp_s[8], tmp_c[8];
                    ভিন্ন acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    প্রতি (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        প্রতি (পূর্ণাঙ্ক j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ চঞ্চল(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(ভিন্ন (*)[8]) acc)
                            : [acc] "m"(*(ধ্রুব ভিন্ন (*)[8]) acc), [sv] "m"(*(ধ্রুব ভিন্ন (*)[8]) tmp_s),
                              [cv] "m"(*(ধ্রুব ভিন্ন (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ চঞ্চল(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(ভিন্ন (*)[8])(x + i1))
                        : [acc] "m"(*(ধ্রুব ভিন্ন (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                প্রতি (; i1 < i1_end; ++i1) {
                    ধ্রুব ভিন্ন * c     = src1_data + i1 * nc;
                    ধ্রুব ভিন্ন * s_row = s + i1 * ncs;
                    ভিন্ন         sumf  = 0.0f;
                    প্রতি (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ চঞ্চল("mova.m.x %0" ::"r"(saved_mask));

    প্রত্যাগম 0;
}
